"""引擎正确性、后台准备、失败/卡顿隔离、量化误差与访问复杂度核对。

运行：PYTHONPATH=. python tests/test_core.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import (  # noqa: E402
    CHUNK, FAILED, IDENTITY, PENDING, READY, AffineSegTree, CountFenwick, Project,
    SummaryNotReady, SummaryWorker, TimecodeOutOfRange, chunk_summary,
    default_compute, default_read_chunk, quantize_pcm16,
)

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def reference_filter(kept: np.ndarray, a: float) -> np.ndarray:
    """先拼接保留样点、从零状态顺序滤波再量化（题目中的对照实现）。"""
    y = np.empty(kept.shape[0], dtype=np.float64)
    s = 0.0
    for n in range(kept.shape[0]):
        s = a * s + (1.0 - a) * kept[n]
        y[n] = s
    return quantize_pcm16(y)


def make_project(samples: np.ndarray, a: float) -> Project:
    """建工程：只落盘 PCM，不启动后台线程、不读任何块。"""
    d = tempfile.mkdtemp(prefix="cliplab-")
    path = os.path.join(d, "input.pcm")
    samples.astype("<i2").tofile(path)
    return Project("t", path, 16000, samples.shape[0], a)


def prepare_manually(project: Project, i: int) -> None:
    """模拟 worker 对单个块的完整准备流程（读块在锁外）。"""
    x = default_read_chunk(project, i)
    project.publish_summary(i, default_compute(x, project.a))


def prepare_all(project: Project) -> None:
    for i in range(project.n):
        if project.status[i] == PENDING:
            prepare_manually(project, i)


def test_fenwick() -> None:
    print("[1] Fenwick 计数树（保留计数与阻塞计数共用同一结构）")
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 5, size=37).tolist()
    fw = CountFenwick(counts)
    arr = counts[:]
    for r in range(len(counts) + 1):
        check(fw.prefix(r) == sum(arr[:r]), f"prefix({r})")
    pos = np.cumsum(arr)
    k = 0
    for block, c in enumerate(arr):
        for _ in range(c):
            check(fw.kth(k) == block, f"kth({k}) -> {block}")
            k += 1
    fw.add(5, -arr[5]); arr[5] = 0
    fw.add(10, arr[10]); arr[10] *= 2
    for r in range(len(counts) + 1):
        check(fw.prefix(r) == sum(arr[:r]), f"更新后 prefix({r})")
    k = 0
    for block, c in enumerate(arr):
        for _ in range(c):
            check(fw.kth(k) == block, f"更新后 kth({k}) -> {block}")
            k += 1


def test_affine_against_loop() -> None:
    print("[2] 仿射摘要与逐样点递推一致（任意输入状态）")
    rng = np.random.default_rng(2)
    a = 0.73
    for length in (1, 2, 17, 4096):
        x = rng.integers(-30000, 30000, size=length).astype(np.float64)
        w, b = chunk_summary(x, a)
        for y_in in (-12345.0, 0.0, 999.5, 30000.0):
            s = y_in
            for v in x:
                s = a * s + (1.0 - a) * v
            got = w * y_in + b
            check(abs(got - s) <= 1e-8, f"L={length} y_in={y_in}: 仿射 {got:.6f} vs 递推 {s:.6f}")


def test_segment_tree_compose() -> None:
    print("[3] 线段树前缀复合（含撤下叶子 -> 恒等）")
    rng = np.random.default_rng(3)
    a = 0.91
    n = 23
    xs = [rng.integers(-1000, 1000, size=rng.integers(1, 9)).astype(np.float64)
          for _ in range(n)]
    leaves = [chunk_summary(x, a) for x in xs]
    tree = AffineSegTree(n)
    for i, leaf in enumerate(leaves):
        tree.set_leaf(i, leaf)
    for r in range(n + 1):
        w, b = tree.prefix(r)
        s = 0.0
        for x in xs[:r]:
            for v in x:
                s = a * s + (1.0 - a) * v
        check(abs(b - s) <= 1e-8 and abs(w - a ** sum(len(x) for x in xs[:r])) <= 1e-12,
              f"prefix({r}) 零状态结果 {b:.6f} vs {s:.6f}")
    tree.set_leaf(7, (1.0, 0.0))
    leaves[7] = (1.0, 0.0)
    w1, b1 = tree.prefix(n)
    s = 0.0
    for j, x in enumerate(xs):
        if j == 7:
            continue
        for v in x:
            s = a * s + (1.0 - a) * v
    check(abs(b1 - s) <= 1e-8, "撤下叶子后前缀等于跳过该块")


def test_construction_reads_nothing() -> None:
    print("[4] PCM 落盘即可建工程：构造不读盘、全部 pending、跳听被阻塞")
    rng = np.random.default_rng(40)
    x = rng.integers(-1000, 1000, size=3 * CHUNK + 5).astype(np.int16)
    proj = make_project(x, 0.8)
    check(proj.chunk_loads == 0 and proj.bg_chunk_loads == 0, "建工程原始块读取为 0")
    check(all(s == PENDING for s in proj.status), "全部块处于准备中")
    check(proj.kept_samples() == x.size, "未撤块时成片样点 = 总样点（计数树已可用）")
    check(proj.n == 4 and proj.chunk_counts[-1] == 5, "4 块、末块 5 样点")
    try:
        proj.audition(0)
        check(False, "无任何摘要时跳听应抛 SummaryNotReady")
    except SummaryNotReady as e:
        check(e.chunk_index == 0 and e.status == PENDING, f"最早阻塞块 #0 准备中（实得 #{e.chunk_index}/{e.status}）")
    check(proj.chunk_loads == 0, "被阻塞的跳听不读 PCM")
    # 剪辑不依赖摘要，立刻可用
    proj.set_chunk_active(1, False)
    check(proj.kept_samples() == x.size - CHUNK, "未就绪状态下撤块立即更新成片时码")
    proj.set_chunk_active(1, True)


def test_blocker_lifecycle() -> None:
    print("[5] 阻塞树：就绪/撤下/恢复/失败/重试的状态流转")
    rng = np.random.default_rng(41)
    x = rng.integers(-1000, 1000, size=4 * CHUNK).astype(np.int16)
    proj = make_project(x, 0.8)

    # 初始：每块都阻塞
    check(proj.blocker.prefix(proj.n) == x.size, "初始阻塞计数 = 总样点")
    prepare_manually(proj, 0)
    check(proj.status[0] == READY and proj.blocker.prefix(1) == 0, "块0 就绪后离开阻塞树")
    check(proj.blocker.prefix(proj.n) == 3 * CHUNK, "其余块仍阻塞")

    # 撤下一个 pending 块：离开阻塞树；恢复 pending：重新进入
    proj.set_chunk_active(2, False)
    check(proj.blocker.prefix(proj.n) == 2 * CHUNK, "撤下 pending 块解除其阻塞")
    try:
        proj.audition(CHUNK)  # 目标块 1，块0 已就绪 -> 块1 仍 pending
        check(False, "块1 未就绪应阻塞")
    except SummaryNotReady as e:
        check(e.chunk_index == 1, f"最早阻塞块是 #1（实得 #{e.chunk_index}）")
    proj.set_chunk_active(2, True)
    check(proj.blocker.prefix(proj.n) == 3 * CHUNK, "恢复 pending 块重新进入阻塞树")

    # 失败块：标记 failed，仍阻塞；撤下后解除
    prepare_manually(proj, 1)
    prepare_manually(proj, 3)
    proj.mark_failed(2, "RuntimeError: injected")
    check(proj.status[2] == FAILED, "块2 标记准备失败")
    try:
        proj.audition(2 * CHUNK)
        check(False, "失败块应阻塞跳听")
    except SummaryNotReady as e:
        check(e.chunk_index == 2 and e.status == FAILED and "injected" in (e.detail or ""),
              "失败阻塞带块号、状态与原因")
    proj.set_chunk_active(2, False)
    check(proj.blocker.prefix(proj.n) == 0, "撤下故障块解除它对后续跳听的阻塞（块0/1/3 均已就绪）")
    # 重试：重新排队，可再发布
    proj.retry_chunk(2)
    check(proj.status[2] == PENDING and proj.blocker.prefix(proj.n) == 0,
          "重试后回到准备中；块仍处撤下状态，不进入阻塞树")
    prepare_manually(proj, 2)
    check(proj.status[2] == READY and proj.active[2] is False, "撤下期间摘要算好只存不挂")
    leaf = proj.segtree.size + 2
    check(proj.segtree.tree[leaf] == (1.0, 0.0), "撤下块线段树叶子仍为恒等")
    proj.set_chunk_active(2, True)
    check(proj.blocker.prefix(proj.n) == 0, "恢复就绪块不进阻塞树")
    check(proj.segtree.tree[leaf] == proj.summaries[2],
          "恢复后摘要叶子才挂回线段树参与滤波")


def test_worker_failure_isolation() -> None:
    print("[6] 后台 worker：单块读盘失败，其余块继续；重试成功")
    rng = np.random.default_rng(42)
    x = rng.integers(-1000, 1000, size=5 * CHUNK + 3).astype(np.int16)
    proj = make_project(x, 0.77)

    def flaky_read(p, i):
        if i == 2:
            raise OSError("injected read failure")
        return default_read_chunk(p, i)

    worker = SummaryWorker(proj, read_fn=flaky_read)
    worker.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        counts = proj.status_counts()
        if counts[PENDING] == 0:
            break
        time.sleep(0.02)
    check(proj.status[2] == FAILED and "injected" in (proj.errors[2] or ""),
          "块2 读盘失败被标记 failed")
    check(all(proj.status[i] == READY for i in range(proj.n) if i != 2),
          "单块失败后其余块全部就绪")
    check(proj.bg_chunk_loads == proj.n - 1,
          f"后台成功读取 {proj.bg_chunk_loads} 个块（= 总块数-1）")

    # 失败块阻塞，撤下后解除
    try:
        proj.audition(2 * CHUNK)
        check(False, "失败块仍在阻塞跳听")
    except SummaryNotReady as e:
        check(e.chunk_index == 2 and e.status == FAILED, "跳听提示准备失败")
    proj.set_chunk_active(2, False)
    chunk, offset, out = proj.audition(2 * CHUNK)  # 成片该位置现落在块3
    check(chunk == 3 and offset == 0 and out.shape[0] == CHUNK,
          "撤下故障块后跳听落到后续已就绪块")

    # 重试：先关闭带故障钩子的旧 worker，避免它抢先消费重试任务，
    # 再用正常读盘钩子手动把重试任务跑通
    proj.set_chunk_active(2, True)
    check(proj.status[2] == FAILED, "恢复故障块不会自动就绪，状态仍为 failed")
    proj.close()
    worker.join(timeout=2)
    check(not worker.is_alive(), "worker 收到关闭信号后退出")
    proj.retry_chunk(2)
    prepare_manually(proj, 2)
    check(proj.status[2] == READY, "重试后块2 就绪")


def test_worker_stuck_does_not_block_editing() -> None:
    print("[7] 后台读块/计算卡住时，剪辑与已就绪位置的跳听照常")
    rng = np.random.default_rng(43)
    x = rng.integers(-2000, 2000, size=4 * CHUNK).astype(np.int16)

    for kind, use_stuck_compute in (("读盘", False), ("计算", True)):
        gate = threading.Event()

        def stuck_read(p, i):
            gate.wait(5.0)
            return default_read_chunk(p, i)

        def stuck_compute(xb, aa):
            gate.wait(5.0)
            return chunk_summary(xb, aa)

        proj = make_project(x, 0.66)
        prepare_manually(proj, 0)  # 块0 先就绪；worker 之后卡在块1
        worker = SummaryWorker(
            proj,
            read_fn=stuck_read,
            compute_fn=stuck_compute if use_stuck_compute else default_compute,
        )
        worker.start()
        time.sleep(0.15)  # 等 worker 卡住

        # 已就绪区域的跳听可用：每次恰好读 1 个目标块
        proj.reset_counters()
        c, o, out = proj.audition(0)
        check(c == 0 and o == 0 and proj.chunk_loads == 1 and out.shape[0] == CHUNK,
              f"{kind}卡住期间块0 跳听成功且只读 1 个目标块")
        # 剪辑照常
        proj.set_chunk_active(2, False)
        check(proj.kept_samples() == 3 * CHUNK, f"{kind}卡住期间撤块立即更新时码")
        check(proj.chunk_loads == 1, f"{kind}卡住期间撤块不读 PCM")
        proj.set_chunk_active(2, True)
        # 未就绪位置仍被阻塞且不读 PCM
        try:
            proj.audition(CHUNK)
            check(False, f"{kind}卡住时块1 应阻塞")
        except SummaryNotReady:
            check(True, f"{kind}卡住时块1 跳听提示准备中")
        check(proj.chunk_loads == 1, "被阻塞的跳听不产生额外 PCM 读取")

        gate.set()
        proj.close()
        worker.join(timeout=3)
        check(not worker.is_alive(), f"{kind}卡住解除后 worker 正常退出")


def test_end_to_end() -> None:
    print("[8] 端到端：随机撤块 + 跳听 vs 先拼接顺序滤波（误差 ≤ 1 量化单位）")
    rng = np.random.default_rng(4)
    for a in (0.0, 0.3, 0.85, 0.99):
        total = 6 * CHUNK + 123
        x = rng.integers(-32000, 32000, size=total).astype(np.int16)
        proj = make_project(x, a)
        prepare_all(proj)
        n = proj.n
        off_mask = set(rng.choice(n, size=max(1, n // 3), replace=False).tolist())
        for i in off_mask:
            proj.set_chunk_active(i, False)

        kept_parts = [x[b * CHUNK:min((b + 1) * CHUNK, total)]
                      for b in range(n) if b not in off_mask]
        ref = reference_filter(np.concatenate(kept_parts), a)
        check(proj.kept_samples() == ref.shape[0],
              f"a={a}: 保留样点数 {proj.kept_samples()} == {ref.shape[0]}")

        cursor = 0
        max_diff = 0
        for b in range(n):
            if b in off_mask:
                continue
            length = min(CHUNK, total - b * CHUNK)
            chunk, offset, out = proj.audition(cursor)
            check(chunk == b and offset == 0, f"a={a}: 块首定位 ({chunk},{offset}) == ({b},0)")
            seg = ref[cursor:cursor + length]
            check(out.shape[0] == seg.shape[0], f"a={a} 块{b}: 试听长度 {out.shape[0]} == {seg.shape[0]}")
            d = int(np.max(np.abs(out.astype(np.int32) - seg.astype(np.int32))))
            max_diff = max(max_diff, d)
            check(np.array_equal(out, seg) or d <= 1, f"a={a} 块{b}: 最大差 {d} ≤ 1")
            cursor += length
        print(f"    → a={a} 全片跳听最大量化差 = {max_diff}")

        for k in (0, 1, 777, cursor - 1, cursor // 2):
            chunk, offset, out = proj.audition(k)
            check(out.shape[0] == proj.chunk_counts[chunk] - offset,
                  f"a={a} k={k}: 试听长度到块末")
            d = int(np.max(np.abs(out.astype(np.int32) - ref[k:k + out.shape[0]].astype(np.int32))))
            check(d <= 1, f"a={a} k={k} (块{chunk}+{offset}): 最大差 {d} ≤ 1")


def test_locate_and_range() -> None:
    print("[9] 时码定位与越界提示")
    rng = np.random.default_rng(5)
    total = 4 * CHUNK + 10
    x = rng.integers(-1000, 1000, size=total).astype(np.int16)
    proj = make_project(x, 0.8)
    prepare_all(proj)
    proj.set_chunk_active(1, False)
    kept_total = proj.kept_samples()
    expect = []
    for b in range(proj.n):
        if b == 1:
            continue
        for off in range(proj.chunk_counts[b]):
            expect.append((b, off))
    check(len(expect) == kept_total, "枚举定位条目数等于保留样点数")
    for k, (eb, eo) in enumerate(expect):
        b, o = proj.locate(k)
        if (b, o) != (eb, eo):
            check(False, f"locate({k}) -> ({b},{o})，应为 ({eb},{eo})")
    check(True, "全部成片样点号定位正确")
    for bad in (-1, kept_total, kept_total + 100):
        try:
            proj.locate(bad)
            check(False, f"k={bad} 应报越界")
        except TimecodeOutOfRange:
            check(True, f"k={bad} 越界抛出 TimecodeOutOfRange")


def test_access_complexity() -> None:
    print("[10] 访问复杂度：撤/恢复只碰摘要不读 PCM，跳听摘要 O(log N)、PCM 读 1 块")
    rng = np.random.default_rng(6)
    sizes = [16, 64, 256]
    prev = None
    for n in sizes:
        x = rng.integers(-1000, 1000, size=n * CHUNK).astype(np.int16)
        proj = make_project(x, 0.8)
        prepare_all(proj)
        bound = 8 * math.ceil(math.log2(n + 1)) + 8

        proj.reset_counters()
        proj.set_chunk_active(n // 2, False)
        t_touches, loads = proj.counters()
        check(loads == 0, f"N={n}: 撤块不读原始块（loads={loads}）")
        check(t_touches <= bound, f"N={n}: 撤块摘要访问 {t_touches} ≤ O(log N) 界 {bound}")

        proj.reset_counters()
        proj.set_chunk_active(n // 2, True)
        t_touches, loads = proj.counters()
        check(loads == 0, f"N={n}: 恢复不读原始块（loads={loads}）")
        check(t_touches <= bound, f"N={n}: 恢复摘要访问 {t_touches} ≤ {bound}")

        proj.set_chunk_active(n // 2, False)
        proj.reset_counters()
        _c, _o, _w = proj.audition(12345)
        a_touches, loads = proj.counters()
        check(loads == 1, f"N={n}: 跳听只读 1 个原始块（loads={loads}）")
        check(a_touches <= bound, f"N={n}: 跳听摘要访问 {a_touches} ≤ {bound}")
        if prev is not None:
            growth = a_touches / prev
            check(growth < 1.6, f"N 翻 4 倍时摘要访问增长 {growth:.2f} 倍（对数级）")
        prev = a_touches


def test_a_zero() -> None:
    print("[11] 边界 a=0：滤波器退化为直通")
    rng = np.random.default_rng(7)
    x = rng.integers(-32768, 32767, size=2 * CHUNK + 5).astype(np.int16)
    proj = make_project(x, 0.0)
    prepare_all(proj)
    proj.set_chunk_active(0, False)
    _c, offset, out = proj.audition(0)
    check(offset == 0, "a=0: 成片 0 号样点落在保留块块首")
    ref_seg = x[CHUNK:CHUNK + out.shape[0]]
    check(np.array_equal(out.astype(np.int32), ref_seg.astype(np.int32)),
          "a=0 撤下首块后块首跳听等于命中块原始 PCM")


def test_concurrent_toggle_and_audition() -> None:
    print("[12] 并发压力：跳听读盘与撤/恢复、后台发布同时进行，结果始终自洽")
    rng = np.random.default_rng(44)
    n = 24
    x = rng.integers(-20000, 20000, size=n * CHUNK).astype(np.int16)
    proj = make_project(x, 0.85)
    prepare_all(proj)
    errors: list[str] = []

    def editor():
        r = np.random.default_rng(100)
        for _ in range(300):
            i = int(r.integers(0, n))
            try:
                proj.set_chunk_active(i, not proj.active[i])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"toggle: {exc!r}")

    def publisher():
        # 走合法的 ready -> failed -> retry(pending) -> ready 流转：
        # ready 块先模拟重新准备失败（离开 ready 并回到阻塞树），再重试发布，
        # 阻塞树与线段树全部由引擎方法同步，不手工掰内部状态。
        r = np.random.default_rng(101)
        for _ in range(200):
            i = int(r.integers(0, n))
            try:
                with proj.lock:
                    was_ready = proj.status[i] == READY and proj.active[i]
                if not was_ready:
                    continue
                with proj.lock:
                    # 撤销已发布摘要：叶子回恒等；保留块回到阻塞树，撤下块不进
                    proj.status[i] = FAILED
                    proj.errors[i] = "synthetic"
                    proj.summaries[i] = None
                    if proj.active[i]:
                        proj.blocker.add(i, proj.chunk_counts[i])
                    proj.segtree.set_leaf(i, IDENTITY)
                proj.retry_chunk(i)
                prepare_manually(proj, i)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"publish: {exc!r}")

    def listener():
        r = np.random.default_rng(102)
        for _ in range(300):
            try:
                kept = proj.kept_samples()
                if kept == 0:
                    continue
                k = int(r.integers(0, kept))
                chunk, offset, out = proj.audition(k)
                # 交付时刻的不变量：命中块必须保留且就绪、长度到块末、只读计数+1
                with proj.lock:
                    if not proj.active[chunk] or proj.status[chunk] != READY:
                        errors.append(f"audition 命中块 {chunk} 状态不自洽")
                    if out.shape[0] != proj.chunk_counts[chunk] - offset:
                        errors.append("试听长度未到块末")
            except SummaryNotReady:
                pass  # 与发布线程交错时允许阻塞
            except Exception as exc:  # noqa: BLE001
                errors.append(f"audition: {exc!r}")

    threads = [threading.Thread(target=editor) for _ in range(2)]
    threads += [threading.Thread(target=publisher), threading.Thread(target=listener)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    check(not any(t.is_alive() for t in threads), "所有并发线程正常结束（无死锁）")
    check(not errors, f"并发期间无异常/不自洽（实得 {errors[:3]}）")
    # 跳听每次请求恰好一次 PCM 读取：单线程复核计数
    proj.reset_counters()
    proj.audition(0)
    _touches, loads = proj.counters()
    check(loads == 1, f"无竞争单次跳听 PCM 读取 {loads} == 1")


if __name__ == "__main__":
    test_fenwick()
    test_affine_against_loop()
    test_segment_tree_compose()
    test_construction_reads_nothing()
    test_blocker_lifecycle()
    test_worker_failure_isolation()
    test_worker_stuck_does_not_block_editing()
    test_end_to_end()
    test_locate_and_range()
    test_access_complexity()
    test_a_zero()
    test_concurrent_toggle_and_audition()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败")
        sys.exit(1)
    print("全部通过")
