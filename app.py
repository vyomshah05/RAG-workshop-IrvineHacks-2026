from rag import RAGBot
import os

def print_sources(retrieved):
    print("\nSources used:")
    if not retrieved:
        print("  (none)")
        return
    for r in retrieved:
        src = os.path.basename(r["source"])
        print(f"  - {src} (chunk {r['chunk_id']}, score={r['score']:.3f})")

def main():
    print("UCI Tour Guide RAG Bot (local + free)")
    print("Type 'exit' to quit.\n")

    bot = RAGBot()

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        ans, retrieved = bot.answer(q)
        print("\nBot:", ans)
        print_sources(retrieved)
        print()

if __name__ == "__main__":
    main()