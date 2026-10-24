from __future__ import annotations

"""TokenMesh High-Performance Latency & Semantic Cache Benchmark.

Executes concurrent and sequential requests against the TokenMesh AI Gateway,
measuring latency distributions (P50, P90, P99) for Cache MISS vs Cache HIT,
sub-millisecond DuckDB HNSW search overhead, and FinOps token economics.

Usage:
    python scripts/benchmark.py [--url http://localhost:8787] [--requests 100] [--concurrency 10]
"""

import argparse
import asyncio
import math
import statistics
import time

import httpx

BENCHMARK_PROMPTS = [
    "Explain the difference between synchronous and asynchronous programming in Python.",
    "What is the mathematical formulation of cross-entropy loss in deep learning?",
    "Write a clean implementation of a LRU Cache in TypeScript with generics.",
    "Describe the ACID properties in distributed transactional databases.",
    "How does the Paxos consensus algorithm achieve agreement across unreliable nodes?",
    "Explain the role of attention heads in the Transformer architecture.",
    "What are the trade-offs between B-Trees and LSM-Trees in database storage engines?",
    "Write an efficient SQL query to find second highest salary across all departments.",
]


def percentile(data: list[float], pct: float) -> float:
    """Calculate percentile from a sorted list of floats."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    model: str,
) -> tuple[float, str, int, float]:
    """Send a single request to /v1/chat/completions and return timing/telemetry."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert system benchmark runner."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=payload, timeout=30.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cache_status = resp.headers.get("x-tokenmesh-cache", "UNKNOWN")
        tokens_saved = int(resp.headers.get("x-tokenmesh-tokens-saved", "0"))
        saved_usd = float(resp.headers.get("x-tokenmesh-saved-usd", "0.0"))
        return elapsed_ms, cache_status, tokens_saved, saved_usd
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms, f"ERROR: {exc}", 0, 0.0


async def run_benchmark(
    base_url: str,
    num_requests: int,
    concurrency: int,
    model: str,
) -> None:
    """Run sequential and concurrent benchmark suite."""
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    print("=" * 72)
    print(" TOKENMESH PERFORMANCE & LATENCY BENCHMARK SUITE")
    print("=" * 72)
    print(f" Target Endpoint:   {api_url}")
    print(f" Model Target:      {model}")
    print(f" Request Count:     {num_requests} passes (Phase 1: MISS, Phase 2: HIT)")
    print(f" Concurrency Level: {concurrency} workers")
    print("=" * 72 + "\n")

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        # Check healthz first
        try:
            h_resp = await client.get(f"{base_url.rstrip('/')}/healthz", timeout=5.0)
            if h_resp.status_code != 200:
                print(f"[!] Gateway at {base_url} returned status {h_resp.status_code}. Aborting.")
                return
        except Exception as exc:
            print(f"[!] Could not connect to TokenMesh gateway at {base_url}: {exc}")
            return

        # ── Phase 1: Priming Cache (Cache MISS) ──────────────────────────
        print("[*] Running Phase 1: Cache Priming (Cold Misses)...")
        miss_latencies: list[float] = []
        phase1_prompts = [
            f"{BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]} [Run ID: {i}-{time.time()}]"
            for i in range(num_requests)
        ]

        sem = asyncio.Semaphore(concurrency)

        async def _worker_miss(p: str) -> tuple[float, str, int, float]:
            async with sem:
                return await send_request(client, api_url, p, model)

        t_start_miss = time.perf_counter()
        results_miss = await asyncio.gather(*[_worker_miss(p) for p in phase1_prompts])
        t_total_miss = time.perf_counter() - t_start_miss

        for lat, _status, _, _ in results_miss:
            miss_latencies.append(lat)

        # ── Phase 2: Querying Cache (Cache HIT) ──────────────────────────
        print("[*] Running Phase 2: Semantic Cache Queries (Hot Hits)...")
        hit_latencies: list[float] = []
        total_tokens_saved = 0
        total_saved_usd = 0.0

        # Query using the exact same prompts that were stored in Phase 1
        async def _worker_hit(p: str) -> tuple[float, str, int, float]:
            async with sem:
                return await send_request(client, api_url, p, model)

        t_start_hit = time.perf_counter()
        results_hit = await asyncio.gather(*[_worker_hit(p) for p in phase1_prompts])
        t_total_hit = time.perf_counter() - t_start_hit

        hit_count = 0
        for lat, status, saved_toks, saved_cost in results_hit:
            hit_latencies.append(lat)
            if status == "HIT":
                hit_count += 1
            total_tokens_saved += saved_toks
            total_saved_usd += saved_cost

    # ── Statistical Calculations ─────────────────────────────────────────
    miss_latencies.sort()
    hit_latencies.sort()

    miss_p50 = percentile(miss_latencies, 50)
    miss_p90 = percentile(miss_latencies, 90)
    miss_p99 = percentile(miss_latencies, 99)
    miss_mean = statistics.mean(miss_latencies) if miss_latencies else 0.0
    miss_rps = num_requests / max(t_total_miss, 0.001)

    hit_p50 = percentile(hit_latencies, 50)
    hit_p90 = percentile(hit_latencies, 90)
    hit_p99 = percentile(hit_latencies, 99)
    hit_mean = statistics.mean(hit_latencies) if hit_latencies else 0.0
    hit_rps = num_requests / max(t_total_hit, 0.001)

    speedup = miss_mean / max(hit_mean, 0.1)
    hit_rate = (hit_count / num_requests) * 100.0 if num_requests else 0.0

    # ── Output Results ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" BENCHMARK RESULTS MATRIX")
    print("=" * 72)
    print("| Metric                      | Cache MISS (Upstream) | Cache HIT (DuckDB VSS) |")
    print("|:----------------------------|----------------------:|-----------------------:|")
    print(f"| Total Requests              | {num_requests:>21d} | {num_requests:>22d} |")
    print(f"| Latency P50 (Median)        | {miss_p50:>18.2f} ms | {hit_p50:>19.2f} ms |")
    print(f"| Latency P90                 | {miss_p90:>18.2f} ms | {hit_p90:>19.2f} ms |")
    print(f"| Latency P99                 | {miss_p99:>18.2f} ms | {hit_p99:>19.2f} ms |")
    print(f"| Mean Latency                | {miss_mean:>18.2f} ms | {hit_mean:>19.2f} ms |")
    print(f"| Throughput (Req/sec)        | {miss_rps:>18.2f} rps| {hit_rps:>19.2f} rps|")
    print("=" * 72)
    print(f" Latency Speedup:       {speedup:.1f}x faster on Cache Hit")
    print(f" Cache Hit Rate:        {hit_rate:.1f}%")
    print(f" Total Tokens Saved:    {total_tokens_saved:,d} tokens")
    print(f" Total Financial Saved: ${total_saved_usd:.4f} USD")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenMesh Latency & FinOps Benchmark Suite")
    parser.add_argument("--url", type=str, default="http://localhost:8787", help="Gateway URL")
    parser.add_argument("--requests", type=int, default=50, help="Number of requests per phase")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Target model")

    args = parser.parse_args()
    asyncio.run(
        run_benchmark(
            base_url=args.url,
            num_requests=args.requests,
            concurrency=args.concurrency,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
