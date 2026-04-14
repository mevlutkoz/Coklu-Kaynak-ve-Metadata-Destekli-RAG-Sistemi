"""Interactive CLI for the multi-source RAG system."""

from dotenv import load_dotenv

load_dotenv()

from src.pipeline import run  # noqa: E402


def main() -> None:
    print("=" * 50)
    print("Multi-Source RAG Sistemi (field-level resolution)")
    print("Çıkmak için 'çıkış' yazın.")
    print("=" * 50 + "\n")

    while True:
        try:
            soru = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSistem kapatılıyor.")
            break

        if not soru:
            continue
        if soru.lower() in ("çıkış", "cikis", "exit", "quit"):
            break

        print("\nİşleniyor...\n")
        try:
            sonuc = run(soru)
            print("=" * 50)
            print(sonuc["cevap"])
            print("=" * 50)
            print("\nResolved facts:")
            for f in sonuc["resolved_facts"]:
                flag = " [OVERRIDE]" if f["overridden"] else ""
                print(
                    f"  {f['field_name']}={f['value']} "
                    f"← {f['source_file']}"
                    f"{' (' + f['effective_date'] + ')' if f.get('effective_date') else ''}"
                    f"{flag}"
                )
            print()
        except Exception as e:
            print(f"Hata oluştu: {e}\n")
            raise


if __name__ == "__main__":
    main()
