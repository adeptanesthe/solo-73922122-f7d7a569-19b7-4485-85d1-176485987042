"""Flask 路由层集成测试：上传即建工程、后台准备、409 阻塞、撤/恢复与跳听。

运行：PYTHONPATH=. python tests/test_web.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.main as m  # noqa: E402
from app.engine import FAILED, PENDING, READY, default_compute, default_read_chunk  # noqa: E402
from app.wavio import build_mono_pcm16_wav  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def wav_bytes(pcm16: np.ndarray, sr: int = 16000) -> bytes:
    return build_mono_pcm16_wav(pcm16.astype("<i2").tobytes(), sr)


class ControllableWorker(m.SummaryWorker):
    """测试用 worker：线程方法空转，准备动作由测试手动触发。"""

    def start(self):
        pass  # 不起线程

    def join(self, timeout=None):
        pass


def prepare_pid(pid: str, *indices: int) -> None:
    p = m.PROJECTS[pid]
    for i in indices:
        x = default_read_chunk(p, i)
        p.publish_summary(i, default_compute(x, p.a))


def make_failed(pid: str, i: int, detail: str) -> None:
    m.PROJECTS[pid].mark_failed(i, detail)


def main() -> None:
    m.DATA_DIR = tempfile.mkdtemp(prefix="cliplab-web-")
    os.makedirs(m.DATA_DIR, exist_ok=True)
    # 上传后不自动起后台线程，全部准备动作由测试控制
    m.SummaryWorker = ControllableWorker
    client = m.app.test_client()

    print("[1] 非法上传被拒绝")
    r = client.post("/upload", data={"a": "0.8", "wav": (io.BytesIO(b"not wav"), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"垃圾文件 -> 400（实得 {r.status_code}）")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(np.zeros(20, dtype="<i2").tobytes())
    r = client.post("/upload", data={"a": "0.8", "wav": (io.BytesIO(buf.getvalue()), "s.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"立体声 -> 400（实得 {r.status_code}）")
    pcm0 = np.zeros(10, dtype="<i2")
    r = client.post("/upload", data={"a": "1.0", "wav": (io.BytesIO(wav_bytes(pcm0)), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"a=1 越界 -> 400（实得 {r.status_code}）")
    r = client.post("/upload", data={"a": "abc", "wav": (io.BytesIO(wav_bytes(pcm0)), "x.wav")},
                    content_type="multipart/form-data")
    check(r.status_code == 400, f"a 非数字 -> 400（实得 {r.status_code}）")

    print("[2] 合法上传：PCM 落盘即建工程，全部 pending、零读取")
    rng = np.random.default_rng(11)
    pcm = rng.integers(-20000, 20000, size=3 * 4096 + 7).astype(np.int16)
    r = client.post("/upload", data={"a": "0.85", "wav": (io.BytesIO(wav_bytes(pcm)), "d.wav")},
                    content_type="multipart/form-data", follow_redirects=False)
    check(r.status_code == 302, f"上传成功 302（实得 {r.status_code}）")
    pid = r.headers["Location"].rstrip("/").split("/")[-1]
    st = client.get(f"/api/p/{pid}/state").get_json()
    check(st["n_chunks"] == 4 and st["kept_samples"] == pcm.size, "4 块、保留全部样点（无需摘要即可剪辑）")
    check(st["chunks"][-1]["samples"] == 7, "末块 7 个样点")
    check(st["summary"] == {"ready": 0, "pending": 4, "failed": 0, "total": 4, "all_ready": False},
          f"摘要初始全部 pending（实得 {st['summary']}）")
    check(m.PROJECTS[pid].chunk_loads == 0, "建工程未读取任何原始块")

    print("[3] 未就绪时跳听 409：提示准备中并定位阻塞块，不读 PCM")
    r = client.get(f"/api/p/{pid}/audition?k=0")
    d = r.get_json()
    check(r.status_code == 409 and d["blocking_chunk"] == 0 and d["blocking_status"] == PENDING,
          f"k=0 无摘要 -> 409 准备中（实得 {r.status_code}/{d.get('blocking_chunk')}）")
    r = client.get(f"/api/p/{pid}/audition.wav?k=0")
    check(r.status_code == 409 and r.get_json()["blocking_status"] == PENDING, "WAV 端点同样 409（播放器无音源）")
    check(m.PROJECTS[pid].chunk_loads == 0, "被阻塞请求不读 PCM")

    print("[4] 逐块准备：准备一块即可跳听该块，目标块直接读 PCM（每次 1 块）")
    prepare_pid(pid, 0)
    st = client.get(f"/api/p/{pid}/state").get_json()
    check(st["summary"]["ready"] == 1 and st["summary"]["pending"] == 3, "状态接口反映 1 就绪 3 准备中")
    r = client.get(f"/api/p/{pid}/audition?k=123")
    meta = r.get_json()
    check(r.status_code == 200 and meta["chunk"] == 0 and meta["offset"] == 123,
          "块0 就绪后块内跳听 200")
    check(meta["chunk_loads"] == 1, f"跳听只读 1 个目标块（实得 {meta['chunk_loads']}）")
    check(meta["out_samples"] == 4096 - 123, "试听长度到块末")
    r = client.get(f"/api/p/{pid}/audition?k=4096")
    check(r.status_code == 409 and r.get_json()["blocking_chunk"] == 1,
          "跨到未就绪块 #1 -> 409，阻塞块号正确")

    print("[5] 撤下/恢复只访问摘要、不读 PCM，成片时码立即更新")
    r = client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": False})
    d = r.get_json()
    check(r.status_code == 200 and d["stats"]["chunk_loads"] == 0,
          f"撤块原始块读取 {d['stats']['chunk_loads']} == 0")
    check(d["stats"]["summary_touches"] <= 40, f"摘要访问 {d['stats']['summary_touches']} 为对数级")
    check(d["state"]["kept_samples"] == pcm.size - 4096, "成片样点数同步减少")
    # 撤下未就绪的块1后，块2 仍是更早的未就绪保留块 -> 阻塞块前进到 #2
    r = client.get(f"/api/p/{pid}/audition?k=4096")
    check(r.status_code == 409 and r.get_json()["blocking_chunk"] == 2,
          "撤下块1后阻塞块变为 #2")
    r = client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": True})
    check(r.get_json()["state"]["kept_samples"] == pcm.size, "恢复后样点数复原")

    print("[6] 全部就绪后的完整跳听与 WAV 内容")
    prepare_pid(pid, 1, 2, 3)
    st = client.get(f"/api/p/{pid}/state").get_json()
    check(st["summary"]["all_ready"] is True, "全部块摘要就绪")
    client.post(f"/api/p/{pid}/chunks/1/toggle", json={"active": False})
    r = client.get(f"/api/p/{pid}/audition?k=4096")
    meta = r.get_json()
    check(r.status_code == 200, f"撤下块1后 k=4096 -> 200（实得 {r.status_code}）")
    check((meta["chunk"], meta["offset"]) == (2, 0), f"定位块#2 偏移0（实得 {meta['chunk']},{meta['offset']}）")
    check(meta["chunk_loads"] == 1 and meta["summary_touches"] <= 32,
          f"读块 {meta['chunk_loads']}、摘要访问 {meta['summary_touches']} 对数级")
    r = client.get(f"/api/p/{pid}/audition.wav?k=4096")
    check(r.status_code == 200 and r.data[:4] == b"RIFF", "WAV 可下载且带 RIFF 头")
    with wave.open(io.BytesIO(r.data)) as wf:
        check(wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 16000,
              "试听 WAV 为单声道 PCM16 同采样率")
        check(wf.getnframes() == 4096, "试听 WAV 帧数 4096")

    print("[7] 准备失败：409 带 failed；撤下故障块解除阻塞；重试后恢复")
    rng2 = np.random.default_rng(12)
    pcm2 = rng2.integers(-20000, 20000, size=3 * 4096).astype(np.int16)
    r = client.post("/upload", data={"a": "0.9", "wav": (io.BytesIO(wav_bytes(pcm2)), "f.wav")},
                    content_type="multipart/form-data", follow_redirects=False)
    pid2 = r.headers["Location"].rstrip("/").split("/")[-1]
    prepare_pid(pid2, 0)
    make_failed(pid2, 1, "OSError: injected disk fault")
    r = client.get(f"/api/p/{pid2}/audition?k=4096")
    d = r.get_json()
    check(r.status_code == 409 and d["blocking_status"] == FAILED and "injected disk fault" in d["error"],
          "失败块 -> 409 且提示准备失败原因")
    r = client.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": False})
    check(r.status_code == 200 and r.get_json()["state"]["kept_samples"] == 2 * 4096,
          "撤下故障块立即更新时码")
    r = client.get(f"/api/p/{pid2}/audition?k=4096")
    check(r.status_code == 409 and r.get_json()["blocking_chunk"] == 2,
          "故障块撤下后不再阻塞，阻塞前进到未就绪的 #2")
    prepare_pid(pid2, 2)
    r = client.get(f"/api/p/{pid2}/audition?k=4096")
    check(r.status_code == 200 and r.get_json()["chunk"] == 2, "撤下故障块后后续跳听可用")
    # 恢复故障块：仍 failed，不会自动挂回；走重试端点重新排队并准备
    client.post(f"/api/p/{pid2}/chunks/1/toggle", json={"active": True})
    r = client.get(f"/api/p/{pid2}/audition?k=4096")
    check(r.status_code == 409 and r.get_json()["blocking_chunk"] == 1, "恢复后故障块仍阻塞")
    r = client.post(f"/api/p/{pid2}/chunks/1/retry")
    check(r.status_code == 200 and r.get_json()["state"]["chunks"][1]["status"] == PENDING,
          "重试端点把块1 重新置为准备中")
    prepare_pid(pid2, 1)
    r = client.get(f"/api/p/{pid2}/audition?k=4096")
    check(r.status_code == 200, "重试准备成功后跳听恢复 200")

    print("[8] 越界时码与未知工程")
    kept = pcm.size - 4096
    for bad in (kept, kept + 10, -5):
        r = client.get(f"/api/p/{pid}/audition?k={bad}")
        check(r.status_code == 400 and "重新选择" in r.get_json()["error"],
              f"k={bad} -> 400 且提示重选")
    r = client.get(f"/api/p/{pid}/audition?k=abc")
    check(r.status_code == 400, "非整数时码 -> 400")
    check(client.get("/api/p/nope/state").status_code == 404, "未知工程 404")

    print("[9] 并发：真实后台线程准备时，前台请求始终拿得到一致状态")
    rng3 = np.random.default_rng(13)
    pcm3 = rng3.integers(-20000, 20000, size=5 * 4096).astype(np.int16)
    # 换回真实 worker 类再上传
    from app.engine import SummaryWorker as RealWorker
    m.SummaryWorker = RealWorker
    r = client.post("/upload", data={"a": "0.8", "wav": (io.BytesIO(wav_bytes(pcm3)), "c.wav")},
                    content_type="multipart/form-data", follow_redirects=False)
    pid3 = r.headers["Location"].rstrip("/").split("/")[-1]
    deadline = time.time() + 10
    while time.time() < deadline:
        st = client.get(f"/api/p/{pid3}/state").get_json()
        if st["summary"]["all_ready"]:
            break
        time.sleep(0.02)
    check(st["summary"]["all_ready"], "真实后台线程把 5 块全部准备完成")
    # 轮询期间并发撤块不报错，最终状态自洽
    for i in range(5):
        rr = client.post(f"/api/p/{pid3}/chunks/{i}/toggle", json={"active": False})
        check(rr.status_code == 200, f"并发撤块 #{i} 200")
    check(client.get(f"/api/p/{pid3}/state").get_json()["kept_samples"] == 0,
          "全部撤下后成片样点为 0")
    m.PROJECTS[pid3].close()

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
