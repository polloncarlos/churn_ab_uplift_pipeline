"""
PA006 — Scoring Pipeline

Executa o pipeline completo de pontuação para uma nova rodada de campanha:
  1. Carrega customer_segments.parquet local (sem dependência de RDS)
  2. Aplica feature engineering (build_features.py)
  3. Churn model → identifica clientes em risco (pool)
  4. Uplift model (T-Learner LR) → estima CATE individual no pool
  5. Classifica 4 perfis e ranqueia por CATE
  6. Gera relatório de negócio + artefatos de saída

Uso:
    python scripts/score_campaign.py
    python scripts/score_campaign.py --top-k 300 --threshold 0.25
"""

import argparse
import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Garante que src/ é importável e CWD é a raiz do projeto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.features.build_features import build_features

# ── Paths ────────────────────────────────────────────────────────────────
CHURN_MODEL_PATH     = ROOT / "models/churn/xgb_churn_calibrated.pkl"
CONTROL_MODEL_PATH   = ROOT / "models/uplift/lr_control.pkl"
TREATMENT_MODEL_PATH = ROOT / "models/uplift/lr_treatment.pkl"
OUTPUT_PARQUET       = ROOT / "data/processed/scoring_output.parquet"
OUTPUT_CSV           = ROOT / "reports/campaign_targets.csv"

# ── Parâmetros de negócio (defaults) ─────────────────────────────────────
CHURN_THRESHOLD = 0.20   # threshold operacional do churn model
CATE_THRESHOLD  = 0.0    # CATE > 0 → campanha ajuda
TOP_K           = 200    # clientes a contatar na campanha
TICKET_MEDIO    = 350
DESCONTO        = 0.10
CUSTO_CONTATO   = 15


# ── Helpers ───────────────────────────────────────────────────────────────

def classify_profile(row, cate_thr, p_ctrl_median):
    high_cate = row["cate_score"] > cate_thr
    high_risk = row["p_control"] >= p_ctrl_median
    if   high_cate and     high_risk: return "Persuadible"
    elif high_cate and not high_risk: return "Sure Thing"
    elif not high_cate and high_risk: return "Lost Cause"
    else:                             return "Sleeping Dog"


def print_section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ── Pipeline principal ────────────────────────────────────────────────────

def run(top_k: int = TOP_K, churn_threshold: float = CHURN_THRESHOLD):

    # ── 1. Feature engineering ───────────────────────────────────────────
    print_section("1. Feature Engineering")
    X, y = build_features(save=False)
    print(f"Dataset: {X.shape[0]} clientes | {X.shape[1]} features")
    print(f"Churn rate (label histórico): {y.mean():.1%}")

    # ── 2. Churn scoring ─────────────────────────────────────────────────
    print_section("2. Churn Scoring")
    churn_model = joblib.load(CHURN_MODEL_PATH)
    churn_scores = churn_model.predict_proba(X)[:, 1]

    scored = X.copy()
    scored["churn_score"] = churn_scores
    scored["churn_label"] = y

    pool = scored[scored["churn_score"] >= churn_threshold].copy()
    print(f"Clientes totais    : {len(scored):,}")
    print(f"Pool (score ≥ {churn_threshold:.2f}) : {len(pool):,} ({len(pool)/len(scored):.1%})")
    print(f"Churn rate no pool : {pool['churn_label'].mean():.1%}")

    # ── 3. Uplift scoring ────────────────────────────────────────────────
    print_section("3. Uplift Scoring (T-Learner LR)")
    model_c = joblib.load(CONTROL_MODEL_PATH)
    model_t = joblib.load(TREATMENT_MODEL_PATH)

    feature_cols = X.columns.tolist()
    X_pool = pool[feature_cols]

    pool["p_control"]   = model_c.predict_proba(X_pool)[:, 1]
    pool["p_treatment"] = model_t.predict_proba(X_pool)[:, 1]
    pool["cate_score"]  = pool["p_control"] - pool["p_treatment"]

    print(f"CATE médio   : {pool['cate_score'].mean():.4f}")
    print(f"CATE std     : {pool['cate_score'].std():.4f}")
    print(f"CATE > 0     : {(pool['cate_score'] > 0).sum()} ({(pool['cate_score'] > 0).mean():.1%})")

    # ── 4. Perfis ────────────────────────────────────────────────────────
    print_section("4. Classificação de Perfis")
    p_ctrl_median = pool["p_control"].median()
    pool["perfil"] = pool.apply(
        classify_profile, axis=1,
        cate_thr=CATE_THRESHOLD, p_ctrl_median=p_ctrl_median
    )

    profile_order = ["Persuadible", "Sure Thing", "Lost Cause", "Sleeping Dog"]
    summary = (
        pool.groupby("perfil")
        .agg(
            n=("cate_score", "count"),
            cate_medio=("cate_score", "mean"),
            churn_score_medio=("churn_score", "mean"),
            churn_real_medio=("churn_label", "mean"),
        )
        .reindex(profile_order)
        .round(3)
    )
    print(summary.to_string())

    # ── 5. Top-K Persuadibles ────────────────────────────────────────────
    print_section(f"5. Top-{top_k} Persuadibles — Análise de Negócio")
    persuadibles = (
        pool[pool["perfil"] == "Persuadible"]
        .sort_values("cate_score", ascending=False)
        .head(top_k)
    )

    n             = len(persuadibles)
    cate_medio    = persuadibles["cate_score"].mean()
    custo         = n * CUSTO_CONTATO
    retencoes_esp = n * cate_medio
    receita_esp   = retencoes_esp * TICKET_MEDIO * (1 - DESCONTO)
    lucro         = receita_esp - custo
    roi           = lucro / custo if custo > 0 else 0

    print(f"Clientes a contatar  : {n}")
    print(f"CATE médio           : {cate_medio:.3f}")
    print(f"Retenções esperadas  : {retencoes_esp:.1f}")
    print(f"Custo da campanha    : R${custo:,.0f}")
    print(f"Receita esperada     : R${receita_esp:,.0f}")
    print(f"Lucro líquido        : R${lucro:,.0f}")
    print(f"ROI direcionado      : {roi:.1%}")
    print()
    print(f"Sleeping Dogs (NÃO contatar): {(pool['perfil'] == 'Sleeping Dog').sum()} clientes")

    # ── 6. Artefatos de saída ────────────────────────────────────────────
    print_section("6. Salvando Artefatos")

    output_cols = [
        "churn_score", "p_control", "p_treatment",
        "cate_score", "perfil", "churn_label",
    ]
    scoring_output = pool[output_cols].copy()
    scoring_output.index.name = "customer_id"
    scoring_output.to_parquet(OUTPUT_PARQUET)
    print(f"Salvo: {OUTPUT_PARQUET.relative_to(ROOT)}  | shape: {scoring_output.shape}")

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    campaign_df = (
        persuadibles[["churn_score", "cate_score", "perfil", "p_control", "p_treatment"]]
        .reset_index()
        .rename(columns={"index": "customer_id"})
        .assign(priority_rank=range(1, len(persuadibles) + 1))
    )
    campaign_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Salvo: {OUTPUT_CSV.relative_to(ROOT)}  | {len(campaign_df)} clientes")

    print_section("Concluído")
    print(f"Pool gerado : {len(pool):,} clientes em risco")
    print(f"Persuadibles: {(pool['perfil'] == 'Persuadible').sum()} identificados")
    print(f"Top-{top_k}     : prontos para campanha em {OUTPUT_CSV.relative_to(ROOT)}")

    return scoring_output, campaign_df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA006 — Scoring Pipeline")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Número de Persuadibles a contatar (padrão: {TOP_K})"
    )
    parser.add_argument(
        "--threshold", type=float, default=CHURN_THRESHOLD,
        help=f"Threshold do churn score para entrar no pool (padrão: {CHURN_THRESHOLD})"
    )
    args = parser.parse_args()
    run(top_k=args.top_k, churn_threshold=args.threshold)
