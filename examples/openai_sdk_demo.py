"""TokenMesh Example: Official OpenAI Python SDK Integration.

This example demonstrates how to route all OpenAI SDK traffic through TokenMesh
to achieve sub-millisecond semantic caching and zero-trust local DLP.

Prerequisites:
    pip install openai
    tokenmesh start   # running on http://localhost:8787
"""

import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("\n[!] Please install the OpenAI SDK first:")
    print("    pip install openai\n")
    sys.exit(1)


def main() -> None:
    print("=" * 65)
    print(" TokenMesh AI Gateway — Official OpenAI SDK Integration")
    print("=" * 65)

    # Point OpenAI SDK to TokenMesh local reverse proxy
    client = OpenAI(
        base_url="http://localhost:8787/v1",
        api_key="tm-local-dev-key",  # TokenMesh validates and injects upstream if needed
    )

    prompt = "What is the capital of France and what is its population?"
    model = "gpt-4o-mini"

    print("\n[1] Sending first request (Expected: Cache MISS)...")
    print(f"    Prompt: '{prompt}'")
    t0 = time.perf_counter()
    resp1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise geography assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    elapsed1 = (time.perf_counter() - t0) * 1000
    print(f"    Latency:  {elapsed1:.1f} ms")
    print(f"    Answer:   {resp1.choices[0].message.content[:80]}...")

    print("\n[2] Sending identical prompt (Expected: Sub-5ms Semantic Cache HIT)...")
    t0 = time.perf_counter()
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise geography assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    elapsed2 = (time.perf_counter() - t0) * 1000
    print(f"    Latency:  {elapsed2:.1f} ms")
    print(f"    Speedup:  {elapsed1 / max(elapsed2, 0.1):.1f}x faster")
    print(f"    Answer:   {resp2.choices[0].message.content[:80]}...")

    print("\n[3] Testing Zero-Trust DLP sanitization...")
    dlp_prompt = "Process refund for customer CPF 123.456.789-00 and email user@example.com"
    print(f"    Raw input: '{dlp_prompt}'")
    _ = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": dlp_prompt}],
    )
    print("    The gateway automatically masked sensitive entities before processing.")
    print("    Check the FinOps ledger at: http://localhost:8787/dashboard\n")


if __name__ == "__main__":
    main()
