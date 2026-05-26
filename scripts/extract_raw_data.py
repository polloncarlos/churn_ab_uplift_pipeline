"""Extract raw data from RDS PostgreSQL and save as parquet to data/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.db_connection import get_engine, load_customer_segments, load_rankers

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Conectando ao RDS...")
    engine = get_engine()

    print("Extraindo vw_customer_segments...", end=" ", flush=True)
    df_segments = load_customer_segments(engine)
    out = RAW_DIR / "customer_segments.parquet"
    df_segments.to_parquet(out, index=False)
    print(f"{len(df_segments):,} linhas → {out.name}")

    print("Extraindo rankers...", end=" ", flush=True)
    df_rankers = load_rankers(engine)
    out = RAW_DIR / "rankers.parquet"
    df_rankers.to_parquet(out, index=False)
    print(f"{len(df_rankers):,} linhas → {out.name}")

    print("\nExtração concluída. Arquivos em data/raw/:")
    for f in sorted(RAW_DIR.glob("*.parquet")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
