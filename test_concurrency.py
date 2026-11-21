"""Hammer the API with overlapping requests that force model swapping.

This reproduces the eviction race: with `registry.get()` outside `gpu_lock`, one
request could evict (and `.to("cpu")`) a model while another was mid-generation
on it, raising "index is on cpu, different from other tensors on cuda:0".

    python test_concurrency.py [--base-url ...] [--api-key ...] [--rounds 4] [--workers 4]
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

HEADERS = {}


# Alternating model families - every adjacent pair forces an evict+load when
# MAX_RESIDENT=1, which is exactly the window the race lived in.
def payloads(base):
    url = f"{base}/v1/speak"
    ref = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
    ref_text = ("Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
                "You blew it! And thanks to you.")
    return [
        ("described", url,
         {"text": "A described voice speaks.", "instruct": "A calm narrator.", "language": "english"}),
        ("built_in", url,
         {"text": "A built-in voice speaks.", "speaker": "ryan", "language": "english"}),
        ("clone", url,
         {"text": "A cloned voice speaks.", "audio": ref, "ref_text": ref_text,
          "language": "english"}),
        ("built_in2", url,
         {"text": "Another built-in voice speaks.", "speaker": "vivian", "language": "english"}),
    ]


def fire(job):
    label, url, body = job
    try:
        r = requests.post(url, json=body, headers=HEADERS, timeout=1200)
        if r.status_code != 200:
            detail = r.text[:200]
            return label, False, f"HTTP {r.status_code}: {detail}"
        return label, True, f"{r.json()['durations'][0]:.1f}s audio"
    except Exception as e:
        return label, False, f"{type(e).__name__}: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:9000")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-key", default=os.environ.get("QWEN_TTS_API_KEY"))
    args = ap.parse_args()
    B = args.base_url.rstrip("/")
    if args.api_key:
        HEADERS["X-API-Key"] = args.api_key

    jobs = payloads(B) * args.rounds
    print(f"firing {len(jobs)} overlapping requests, {args.workers} at a time\n")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (label, ok, note) in enumerate(pool.map(fire, jobs), 1):
            print(f"  [{'PASS' if ok else 'FAIL'}] {i:3}/{len(jobs)} {label:10} {note}")
            if not ok:
                failures.append((label, note))

    print("\n" + "=" * 72)
    if failures:
        print(f"  {len(failures)}/{len(jobs)} FAILED")
        for label, note in failures[:10]:
            print(f"    {label}: {note}")
    else:
        print(f"  all {len(jobs)} concurrent requests succeeded - no eviction race")

    # Did the batcher actually merge anything? avg_batch_size stuck at 1.0 while
    # requests genuinely overlapped means batching is not engaging.
    try:
        load = requests.get(f"{B}/health", timeout=30).json().get("load", {})
        print(f"\n  avg batch size   : {load.get('avg_batch_size')}")
        print(f"  max batch seen   : {load.get('max_batch_size_seen')}")
        print(f"  queue wait p95   : {load.get('queue_wait_p95_seconds')}s")
        print(f"  completed/failed : {load.get('completed')}/{load.get('failed')}")
        if load.get("max_batch_size_seen", 0) <= 1 and args.workers > 1:
            print("  NOTE batching never merged - check QWEN_TTS_MAX_BATCH > 1")
    except Exception as e:
        print(f"  (could not read /health: {e})")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
