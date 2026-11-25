"""分块剪辑引擎（后台逐块准备摘要版）。

核心思想
========
录音按 ``CHUNK`` 个样点分块，PCM 一落盘即可建工程：建工程时**不读任何原始块**，
每块的低通仿射摘要由后台 :class:`SummaryWorker` 逐块计算。剪辑师可以立刻开始
撤下/恢复块，跳听只依赖“已经备好摘要的块”。

在任意一组撤下决策与一组摘要状态之上，支持两个亚线性操作：

1. **成片时码定位**：成片第 k 个保留样点，对应哪个原块、块内偏移多少？
2. **接缝跳听**：从成片第 k 个样点到其所在块块末的低通滤波结果。

数据结构
--------
* :class:`CountFenwick`（计数树 A）—— 每块保留样点数（撤下=0），支持单点增减、
  前缀和、第 k 个保留位置查询，全部 O(log N)。
* :class:`CountFenwick`（计数树 B，“阻塞树”）—— 每块在**保留且摘要未就绪**
  （准备中/准备失败）时计入块样点数，就绪或撤下即减为 0。跳听前用一次前缀和 +
  一次 kth(0) 即可在 O(log N) 内找到最早的阻塞块。
* :class:`AffineSegTree` —— 一阶低通 ``y[n] = a*y[n-1] + (1-a)*x[n]`` 对一块
  样点的作用是关于输入状态的仿射变换 ``y_out = w*y_in + b``。线段树叶子只放
  **已就绪且保留**块的摘要，其余叶子是恒等 (1, 0)。

状态约定
--------
每块有两个互相独立的状态：

* ``active``：保留 / 撤下（剪辑师决策，随时切换）；
* ``status``：``pending``（准备中）/ ``ready``（就绪）/ ``failed``（准备失败）。

撤下/恢复只改三棵树的 O(log N) 个节点，不读 PCM；摘要在撤下期间算好也只存不挂，
恢复时才把叶子换进线段树参与滤波。跳听先查阻塞树：目标块之前若还有保留但未就绪
的块，抛 :class:`SummaryNotReady`（由网页提示准备中/准备失败并收起播放器）；
否则线段树前缀查询恢复入块状态，只读 1 个目标块的 PCM 逐样点递推。
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

CHUNK = 4096

# 块摘要状态
PENDING = "pending"
READY = "ready"
FAILED = "failed"


class TimecodeOutOfRange(Exception):
    """成片样点号越界。"""


class SummaryNotReady(Exception):
    """跳听被“保留但摘要未就绪”的块阻塞。"""

    def __init__(self, chunk_index: int, status: str, detail: str | None = None):
        self.chunk_index = chunk_index
        self.status = status
        self.detail = detail
        verb = "摘要准备失败" if status == FAILED else "摘要仍在准备中"
        tail = f"：{detail}" if detail else ""
        super().__init__(
            f"块 #{chunk_index} {verb}{tail}，跳听被阻塞；可撤下该块解除阻塞，"
            f"或等待准备完成（失败可重试）"
        )


def compose(g: tuple[float, float], h: tuple[float, float]) -> tuple[float, float]:
    """先做 g 再做 h 的复合仿射变换。

    g(y) = wg*y + bg, h(y) = wh*y + bh
    h(g(y)) = (wh*wg)*y + (wh*bg + bh)
    """
    wg, bg = g
    wh, bh = h
    return (wh * wg, wh * bg + bh)


IDENTITY = (1.0, 0.0)


class CountFenwick:
    """块计数的 Fenwick 树（1 基内部存储），所有操作 O(log N)。"""

    def __init__(self, counts: list[int]):
        n = len(counts)
        self.n = n
        bit = [0] * (n + 1)
        for i, v in enumerate(counts, start=1):
            bit[i] += v
            j = i + (i & -i)
            if j <= n:
                bit[j] += bit[i]
        self.bit = bit
        self.touches = 0

    def add(self, i: int, delta: int) -> None:
        """0 基位置 i 单点增减。"""
        i += 1
        n = self.n
        bit = self.bit
        while i <= n:
            bit[i] += delta
            self.touches += 1
            i += i & -i

    def prefix(self, r: int) -> int:
        """前 r 个块 [0, r) 的计数总和。"""
        s = 0
        bit = self.bit
        i = r
        while i > 0:
            s += bit[i]
            self.touches += 1
            i -= i & -i
        return s

    def kth(self, k: int) -> int:
        """第 k（0 基）个计数位置所在的块号。"""
        bit = self.bit
        n = self.n
        idx = 0
        step = 1 << (n.bit_length() - 1)
        while step:
            nxt = idx + step
            if nxt <= n and bit[nxt] <= k:
                idx = nxt
                k -= bit[nxt]
            self.touches += 1
            step >>= 1
        return idx


class AffineSegTree:
    """块仿射摘要的线段树，叶子在 [size, 2*size)。"""

    def __init__(self, n: int):
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        self.n = n
        tree = [IDENTITY] * (2 * size)
        for i in range(size - 1, 0, -1):
            tree[i] = compose(tree[2 * i], tree[2 * i + 1])
        self.tree = tree
        self.touches = 0

    def set_leaf(self, i: int, summary: tuple[float, float]) -> None:
        """把块 i 的摘要换成 summary（撤下 -> 恒等，恢复/发布 -> 滤波摘要）。"""
        t = self.tree
        p = self.size + i
        t[p] = summary
        self.touches += 1
        p >>= 1
        while p:
            t[p] = compose(t[2 * p], t[2 * p + 1])
            self.touches += 1
            p >>= 1

    def prefix(self, r: int) -> tuple[float, float]:
        """前 r 个块 [0, r) 顺序复合后的仿射变换；r=0 时为恒等。"""
        t = self.tree
        l = self.size
        rr = self.size + r
        left_acc = IDENTITY
        right_acc = IDENTITY
        while l < rr:
            if l & 1:
                left_acc = compose(left_acc, t[l])
                self.touches += 1
                l += 1
            if rr & 1:
                rr -= 1
                right_acc = compose(t[rr], right_acc)
                self.touches += 1
            l >>= 1
            rr >>= 1
        return compose(left_acc, right_acc)


def chunk_summary(x: np.ndarray, a: float) -> tuple[float, float]:
    """一块样点 x 对一阶低通的仿射摘要 (w, b)：y_out = w*y_in + b。

    递推 L 步后 ``y_out = a^L * y_in + (1-a) * sum_n a^(L-1-n) x[n]``，
    全程 float64、中途不取整，与逐样点递推共用同一公式。
    """
    L = x.shape[0]
    if L == 0:
        return IDENTITY
    xf = x.astype(np.float64)
    w = float(a) ** L
    # a^(L-1), a^(L-2), ..., a^0 ；L <= 4096，整数指数在 float64 中精确
    powers = np.float64(a) ** np.arange(L - 1, -1, -1, dtype=np.float64)
    b = float((1.0 - a) * np.dot(powers, xf))
    return (w, b)


def quantize_pcm16(y: np.ndarray) -> np.ndarray:
    """float64 滤波结果量化回 PCM16（四舍五入 + 夹幅）。"""
    return np.clip(np.rint(y), -32768.0, 32767.0).astype("<i2")


@dataclass
class ChunkInfo:
    index: int
    samples: int          # 该块原始样点数
    active: bool
    status: str           # pending / ready / failed
    error: str | None     # 失败原因（仅 failed）
    start_sample: int     # 在原始录音中的起始样点号


class Project:
    """一个上传录音的剪辑工程。

    构造时只根据样点总数推算分块元数据，**不读取 PCM**；摘要由后台 worker
    逐块发布。所有会改树/读树的方法都在 ``self.lock`` 下执行；原始块读取
    （``_read_chunk``）在锁外进行，因此后台读盘或计算卡住不会挡住剪辑与跳听。
    """

    def __init__(self, project_id: str, pcm_path: str, sample_rate: int,
                 total_samples: int, a: float):
        self.id = project_id
        self.pcm_path = pcm_path
        self.sample_rate = sample_rate
        self.total_samples = total_samples
        self.a = float(a)
        self.n = math.ceil(total_samples / CHUNK)

        # 分块元数据直接由总样点数推出，不需要读盘
        self.chunk_counts = [
            min(CHUNK, total_samples - i * CHUNK) for i in range(self.n)
        ]
        self.active = [True] * self.n
        self.status = [PENDING] * self.n
        self.errors: list[str | None] = [None] * self.n
        self.summaries: list[tuple[float, float] | None] = [None] * self.n

        # 计数树 A：保留样点数（初始全部保留）
        self.fenwick = CountFenwick(self.chunk_counts)
        # 计数树 B：保留且摘要未就绪的块（初始全部 pending，计数与 A 相同）
        self.blocker = CountFenwick(self.chunk_counts)
        # 仿射线段树：初始全部恒等叶子，摘要发布/恢复时再换
        self.segtree = AffineSegTree(self.n)

        # 后台任务队列：按块号顺序准备，失败块重试时插到队首
        self._queue: deque[int] = deque(range(self.n))
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        # 跳听读盘与撤/恢复、摘要发布之间的状态一致性锁：只包住“取快照+读盘”
        # 这段，后台 worker 的实际计算不持有它，因此计算卡住不会阻塞跳听。
        self.read_lock = threading.Lock()

        # 访问计数（只允许持锁修改）：前台/后台分开统计
        self.chunk_loads = 0        # 跳听等前台操作读取的原始块数
        self.bg_chunk_loads = 0     # 后台摘要读取的原始块数

    # ---------- 原始样点：只允许按块随机读取，且在工程锁外调用 ----------
    def _read_chunk(self, i: int) -> np.ndarray:
        start = i * CHUNK
        count = min(CHUNK, self.total_samples - start)
        with open(self.pcm_path, "rb") as f:
            f.seek(start * 2)  # PCM16 -> 每样点 2 字节
            raw = f.read(count * 2)
        return np.frombuffer(raw, dtype="<i2")

    # ---------- 计数快照（路由层用于统计单次操作的访问量） ----------
    def counters(self) -> tuple[int, int]:
        """返回 (摘要/树节点访问数, 前台原始块读取数)。"""
        with self.lock:
            touches = self.fenwick.touches + self.blocker.touches + self.segtree.touches
            return touches, self.chunk_loads

    def reset_counters(self) -> None:
        with self.lock:
            self.fenwick.touches = 0
            self.blocker.touches = 0
            self.segtree.touches = 0
            self.chunk_loads = 0
            self.bg_chunk_loads = 0

    # ---------- 工程信息 ----------
    def kept_samples(self) -> int:
        with self.lock:
            return self.fenwick.prefix(self.n)

    def status_counts(self) -> dict[str, int]:
        with self.lock:
            return {
                READY: sum(1 for s in self.status if s == READY),
                PENDING: sum(1 for s in self.status if s == PENDING),
                FAILED: sum(1 for s in self.status if s == FAILED),
            }

    def list_chunks(self) -> list[ChunkInfo]:
        with self.lock:
            return [
                ChunkInfo(
                    index=i,
                    samples=self.chunk_counts[i],
                    active=self.active[i],
                    status=self.status[i],
                    error=self.errors[i],
                    start_sample=i * CHUNK,
                )
                for i in range(self.n)
            ]

    # ---------- 撤下 / 恢复：只访问三棵树的摘要，不读 PCM ----------
    def set_chunk_active(self, i: int, active: bool) -> None:
        # 与跳听“快照+读盘”段互斥，保证跳听拿到的剪法快照和读回的 PCM 一致；
        # 本方法自身仍只读树、不碰 PCM。后台 worker 的计算不持有 read_lock。
        with self.read_lock:
            with self.lock:
                if not (0 <= i < self.n):
                    raise IndexError(f"块号 {i} 越界，共 {self.n} 块")
                if self.active[i] == active:
                    return
                self.active[i] = active
                delta = self.chunk_counts[i] if active else -self.chunk_counts[i]
                self.fenwick.add(i, delta)
                if self.status[i] != READY:
                    # 未就绪块在阻塞树里跟随保留状态进出
                    self.blocker.add(i, delta)
                if self.status[i] == READY:
                    # 就绪块：恢复挂回摘要叶子，撤下换成恒等
                    self.segtree.set_leaf(i, self.summaries[i] if active else IDENTITY)
                # 未就绪块的线段树叶子始终是恒等，无需改动

    # ---------- 后台 worker 回调（全部持锁，只碰 O(log N) 节点） ----------
    def pop_job(self) -> int | None:
        """取出下一个待准备块号；队列空时唤醒事件复位。"""
        with self.lock:
            if self._queue:
                return self._queue.popleft()
            self.wake.clear()
            return None

    def publish_summary(self, i: int, summary: tuple[float, float]) -> None:
        """后台摘要算好后发布：撤下期间只存不挂，恢复后才参与滤波。"""
        # 与跳听听读盘互斥，避免“跳听快照在前、叶子发布在后”的交错；
        # 注意 worker 的读盘与计算在 read_lock 之外，卡住不会阻塞前台。
        with self.read_lock:
            with self.lock:
                if self.status[i] != PENDING:
                    return  # 已被重试/关闭等流程接管，丢弃过期结果
                self.status[i] = READY
                self.errors[i] = None
                self.summaries[i] = summary
                if self.active[i]:
                    self.blocker.add(i, -self.chunk_counts[i])
                    self.segtree.set_leaf(i, summary)

    def mark_failed(self, i: int, detail: str) -> None:
        """单块准备失败：保留在阻塞树里，其余块不受影响、继续准备。"""
        with self.lock:
            if self.status[i] != PENDING:
                return
            self.status[i] = FAILED
            self.errors[i] = detail

    def retry_chunk(self, i: int) -> None:
        """失败块重新入队（队首优先），撤下状态也可以重试。"""
        with self.lock:
            if not (0 <= i < self.n):
                raise IndexError(f"块号 {i} 越界，共 {self.n} 块")
            if self.status[i] != FAILED:
                return
            self.status[i] = PENDING
            self.errors[i] = None
            self._queue.appendleft(i)
            self.wake.set()

    def close(self) -> None:
        with self.lock:
            self.stop_event.set()
            self.wake.set()

    # ---------- 成片时码定位 ----------
    def locate(self, k: int) -> tuple[int, int]:
        """成片 0 基样点号 k -> (原块号, 块内偏移)。调用方持锁。"""
        kept = self.fenwick.prefix(self.n)
        if not (0 <= k < kept):
            raise TimecodeOutOfRange(
                f"成片样点号 {k} 越界：当前保留 {kept} 个样点，"
                f"有效范围 0..{kept - 1}"
            )
        chunk = self.fenwick.kth(k)
        before = self.fenwick.prefix(chunk)
        offset = k - before
        return chunk, offset

    def _first_blocker(self, target_chunk: int) -> int | None:
        """目标块之前（含目标块自身）最早的“保留且未就绪”块号；没有则 None。"""
        if self.blocker.prefix(target_chunk) > 0:
            return self.blocker.kth(0)
        if self.status[target_chunk] != READY:
            return target_chunk
        return None

    # ---------- 跳听：摘要访问 O(log N)，最多只读 1 个目标块 PCM ----------
    def audition(self, k: int) -> tuple[int, int, np.ndarray]:
        """返回 (原块号, 块内偏移, 从 k 到块末的 PCM16 滤波结果)。

        目标块之前仍保留的块若有任一摘要未就绪，抛 :class:`SummaryNotReady`。
        ``read_lock`` 只覆盖前台的“取剪法快照 + 读目标块”这一段，与撤/恢复、
        摘要发布互斥，因此一次请求内块状态与读回的 PCM 必然一致，且恰好读一次
        PCM；后台 worker 的读盘与计算不持有该锁，卡住也不会阻塞这里。
        """
        with self.read_lock:
            with self.lock:
                chunk, offset = self.locate(k)
                blocker = self._first_blocker(chunk)
                if blocker is not None:
                    raise SummaryNotReady(
                        blocker, self.status[blocker], self.errors[blocker]
                    )
                # 入块前滤波器状态：前缀块复合仿射作用于零状态 y[-1]=0
                _, state_in = self.segtree.prefix(chunk)
                aa = self.a

            # 锁外读目标块：read_lock 挡住前台剪法变更，后台计算不持此锁
            x = self._read_chunk(chunk)

            with self.lock:
                self.chunk_loads += 1

        xf = x.astype(np.float64)
        one_minus_a = 1.0 - aa
        out = np.empty(xf.shape[0], dtype=np.float64)
        s = float(state_in)
        # 块内逐样点递推，不取整
        for n in range(xf.shape[0]):
            s = aa * s + one_minus_a * xf[n]
            out[n] = s
        return chunk, offset, quantize_pcm16(out[offset:])


def default_read_chunk(project: Project, i: int) -> np.ndarray:
    """后台 worker 默认读块钩子（锁外执行）。"""
    x = project._read_chunk(i)
    with project.lock:
        project.bg_chunk_loads += 1
    return x


def default_compute(x: np.ndarray, a: float) -> tuple[float, float]:
    """后台 worker 默认计算钩子（锁外执行）。"""
    return chunk_summary(x, a)


class SummaryWorker(threading.Thread):
    """后台逐块准备摘要的守护线程。

    读盘与计算都在工程锁外执行，因此单个块读盘/计算卡住（或被测试钩子阻塞）
    时，剪辑与“已就绪位置”的跳听照常可用。单块抛异常只标记该块 failed，
    队列里其余块继续处理。``read_fn`` / ``compute_fn`` 可替换以注入失败/卡顿。
    """

    def __init__(self, project: Project, *, read_fn=default_read_chunk,
                 compute_fn=default_compute, per_chunk_delay: float = 0.0,
                 name: str | None = None):
        super().__init__(daemon=True, name=name or f"summary-{project.id}")
        self.project = project
        self.read_fn = read_fn
        self.compute_fn = compute_fn
        self.per_chunk_delay = per_chunk_delay

    def run(self) -> None:
        p = self.project
        while not p.stop_event.is_set():
            i = p.pop_job()
            if i is None:
                p.wake.wait(timeout=0.25)
                continue
            try:
                if self.per_chunk_delay > 0:
                    if p.stop_event.wait(self.per_chunk_delay):
                        return
                x = self.read_fn(p, i)
                summary = self.compute_fn(x, p.a)
            except Exception as exc:  # 单块失败不影响其余块
                p.mark_failed(i, f"{type(exc).__name__}: {exc}")
            else:
                p.publish_summary(i, summary)
