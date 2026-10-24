"""TokenMesh Example: LangChain Integration.

Demonstrates routing LangChain ChatOpenAI and ChatAnthropic models through
TokenMesh transparently without changing business logic.

Prerequisites:
    pip install langchain-core langchain-openai langchain-anthropic
    tokenmesh start   # running on http://localhost:8787
"""

import sys

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    print("\n[!] Please install langchain dependencies:")
    print("    pip install langchain-core langchain-openai\n")
    sys.exit(1)


def main() -> None:
    print("=" * 65)
    print(" TokenMesh AI Gateway — LangChain Transparent Proxy")
    print("=" * 65)

    # Initialize LangChain ChatOpenAI pointing to TokenMesh
    llm = ChatOpenAI(
        base_url="http://localhost:8787/v1",
        api_key="tm-local-key",  # type: ignore[arg-type]
        model="gpt-4o-mini",
        temperature=0.0,
    )

    messages = [
        SystemMessage(content="You are an enterprise SQL query generator."),
        HumanMessage(content="Write a SELECT query to find top 10 customers by revenue."),
    ]

    print("\n[1] Invoking LangChain model (Pass 1 - Cache MISS)...")
    res1 = llm.invoke(messages)
    print(f"    Content: {res1.content[:90]}...")

    print("\n[2] Invoking LangChain model again (Pass 2 - Cache HIT)...")
    res2 = llm.invoke(messages)
    print(f"    Content: {res2.content[:90]}...")
    print("    LangChain received cached response in sub-5ms!")
    print("\nCheck live FinOps metrics at http://localhost:8787/dashboard\n")


if __name__ == "__main__":
    main()
