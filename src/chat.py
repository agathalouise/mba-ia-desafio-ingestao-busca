import sys
try:
    from search import search_prompt
except ModuleNotFoundError:
    from src.search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("=" * 60)
    print("Chat CLI - Ingestão e Busca Semântica (LangChain + pgVector)")
    print("Digite sua pergunta ou 'sair' para encerrar.")
    print("=" * 60)

    while True:
        try:
            question = input("\nFaça sua pergunta: ").strip()
            if not question:
                continue

            if question.lower() in ("sair", "exit", "quit", "q"):
                print("Encerrando o chat. Até logo!")
                break

            response = chain.invoke(question)

            print(f"\nPERGUNTA: {question}")
            print(f"RESPOSTA: {response.strip()}")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat. Até logo!")
            break
        except Exception as exc:
            print(f"Erro ao processar pergunta: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()