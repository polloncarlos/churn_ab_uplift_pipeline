"""Feature engineering pipeline for churn prediction (PA006)."""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

CHURN_THRESHOLD = 90

# Colunas excluídas do X de treino
DROP_COLS = [
    "recency_days",     # define o target — leakage
    "cluster_id",       # derivado dos dados — leakage indireto
    "cluster_label",
    "cluster_order",
    "net_revenue",      # multicolinear com gross_revenue
]

# Features com distribuição assimétrica que se beneficiam de log
LOG_FEATURES = [
    "gross_revenue",
    "total_items",
    "revenue_velocity",
    "return_value",
    "basket_size",
    "items_velocity",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["churn"] = (df["recency_days"] > CHURN_THRESHOLD).astype(int)
    return df


def filter_modeling_population(df: pd.DataFrame) -> pd.DataFrame:
    # Remove one-time buyers (ativação, não retenção) e quasi one-time buyers
    # (2 compras no mesmo dia, lifetime=0 — sem relação temporal com a marca)
    mask = (df["frequency"] >= 2) & (df["customer_lifetime_days"] > 0)
    filtered = df[mask].copy()
    print(f"Filtro aplicado: {len(df)} → {len(filtered)} clientes "
          f"({len(df) - len(filtered)} removidos)")
    return filtered


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG_FEATURES:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    return df


def build_features(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o pipeline completo de feature engineering.

    Returns:
        X: features de treino
        y: target (churn)
    """
    df = pd.read_parquet(RAW_DIR / "customer_segments.parquet")

    df = build_target(df)
    df = filter_modeling_population(df)
    df = apply_log_transforms(df)
    df = df.set_index("customer_id")

    y = df["churn"]
    X = df.drop(columns=DROP_COLS + ["churn"])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(PROCESSED_DIR / "X_churn.parquet", index=True)
        y.to_frame().to_parquet(PROCESSED_DIR / "y_churn.parquet", index=True)
        print(f"Salvo: X {X.shape}, y {y.shape} → data/processed/")

    return X, y


if __name__ == "__main__":
    X, y = build_features()
    print(f"\nFeatures ({X.shape[1]}):")
    print(list(X.columns))
    print(f"\nChurn rate: {y.mean():.1%}")
