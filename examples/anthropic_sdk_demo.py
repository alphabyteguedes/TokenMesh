"""TokenMesh Example: Official Anthropic Python SDK Integration.

This example demonstrates how to route Anthropic SDK traffic through TokenMesh
to achieve semantic caching and FinOps savings on Claude 3.5 & 3.7 models.

Prerequisites:
    pip install anthropic
    tokenmesh start   # running on http://localhost:8787
"""

import sys
import time

try:
    from anthropic import Anthropic
except ImportError:
    print("\n[!] Please install the Anthropic SDK first:")
    print("    pip install anthropic\n")
    sys.exit(1)


def main() -> None:
    print("=" * 65)
    print(" TokenMesh AI Gateway — Official Anthropic SDK Integration")
    print("=" * 65)

    # Point Anthropic SDK to TokenMesh local reverse proxy
    client = Anthropic(
        base_url="http://localhost:8787",
        api_key="tm-local-dev-key",
    )

    prompt = "Explain quantum computing in three short bullet points."
    model = "claude-3-5-sonnet-latest"

    print("\n[1] Sending first request (Expected: Cache MISS)...")
    print(f"    Prompt: '{prompt}'")
    t0 = time.perf_counter()
    resp1 = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed1 = (time.perf_counter() - t0) * 1000
    first_block = resp1.content[0].text if resp1.content else ""
    print(f"    Latency:  {elapsed1:.1f} ms")
    print(f"    Answer:   {first_block[:80]}...")

    print("\n[2] Sending semantically equivalent request (Expected: Cache HIT)...")
    t0 = time.perf_counter()
    resp2 = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed2 = (time.perf_counter() - t0) * 1000
    second_block = resp2.content[0].text if resp2.content else ""
    print(f"    Latency:  {elapsed2:.1f} ms")
    print(f"    Speedup:  {elapsed1 / max(elapsed2, 0.1):.1f}x faster")
    print(f"    Answer:   {second_block[:80]}...")

    print("\n[3] Check the live ledger at: http://localhost:8787/dashboard\n")


if __name__ == "__main__":
    main()
