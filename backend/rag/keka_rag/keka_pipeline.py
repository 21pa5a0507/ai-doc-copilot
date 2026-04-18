from rag.keka_rag.service import initialize_keka_service


def main():
    keka_service = initialize_keka_service()
    rag = keka_service.rag_chain

    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = rag(query, debug=False)
        print("\n💡 Answer:\n", answer)


if __name__ == "__main__":
    main()
