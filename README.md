# PA006 — Churn Prediction + A/B Testing + Uplift Modeling

Projeto end-to-end de ciência de dados seguindo o framework CRISP-DS. Identifica clientes com risco de churn em um e-commerce, valida estratégias de retenção com rigor estatístico (A/B Testing) e descobre para quais clientes a intervenção realmente gera impacto (Uplift Modeling).

**Autor:** Carlos Pollon | [GitHub](https://github.com/polloncarlos) | [Portfólio](https://polloncarlos.github.io)
**Período:** Jun 2026 –

---

## Contexto de negócio

**Empresa simulada:** e-commerce B2C de médio porte, 4.300 clientes ativos, ticket médio de R$ 350. O time de CRM identificou que clientes sem compra há 90 dias raramente retornam espontaneamente — e o custo de reativação cresce quanto mais tempo passa.

**A dor:** o time de marketing dispõe de um budget fixo para campanhas de retenção (cupom de 10% + e-mail personalizado, custo de R$ 15 por contato). O problema é que contatar todos os clientes "em risco" esgota o budget e dilui o impacto. Precisam saber **onde concentrar esforço**.

**As duas perguntas que o projeto responde:**

1. Quais clientes têm maior probabilidade de churnar nos próximos 90 dias?
2. Para quais desses clientes a campanha realmente **muda o comportamento** (Persuadables), versus clientes que converteriam de qualquer forma (Sure Things) ou que não responderiam de jeito nenhum (Lost Causes)?

**Premissas da campanha (simuladas):**

| Parâmetro | Valor |
|-----------|-------|
| Custo por contato | R$ 15 |
| Taxa de retenção da campanha | 30% |
| Receita média por cliente retido | R$ 350 |
| Retorno líquido por churner retido | R$ 90 |

**Escolha do threshold — lógica de negócio:**

Falso negativo (FN) custa R$ 350 em receita perdida; falso positivo (FP) custa R$ 15 em campanha desperdiçada. Relação de custo FN/FP = 7×. Isso justifica um threshold conservador (0.20) que maximiza recall em detrimento de precision, capturando 84.8% dos churners com ROI líquido estimado de R$ 30.5k por ciclo de campanha sobre a base de 2.758 clientes elegíveis.

---

## Arquitetura — 3 camadas

```
Camada 1 — Churn Prediction
  Dados do RDS → feature engineering → XGBoost calibrado → score P(churn) por cliente

Camada 2 — A/B Testing
  Score de churn → divisão grupo controle/tratamento → simulação de campanha
  → teste de hipótese (scipy.stats) → significância estatística

Camada 3 — Uplift Modeling
  Dados do experimento → T-Learner (causalml) → score de uplift por cliente
  → segmentação: Persuadables / Sure Things / Lost Causes / Sleeping Dogs
  → AUUC + uplift curve → recomendação de alvo da campanha
```

---

## Stack técnica

| Camada | Bibliotecas |
|--------|------------|
| Dados | pandas, numpy, sqlalchemy, psycopg2 |
| Feature engineering | pandas, scikit-learn (Pipeline, ColumnTransformer) |
| Modelagem churn | xgboost, scikit-learn, optuna |
| Calibração | scikit-learn (CalibratedClassifierCV) |
| A/B Testing | scipy.stats, statsmodels |
| Uplift Modeling | causalml |
| Visualização | matplotlib, seaborn, plotly |
| Ambiente | python-dotenv, jupyter |

---

## Origem dos dados

Os dados vêm do **PA005 — Customer Value Segmentation**, projeto anterior com dados já tratados e segmentados em produção.

**Fonte:** RDS PostgreSQL (AWS) — banco `ecommerce`

| Tabela | Conteúdo |
|--------|----------|
| `customers` | Cadastro de clientes (customer_id, signup_date, país) |
| `transactions` | Histórico de compras (customer_id, invoice_date, amount, quantity, product) |
| `customer_features` | 17 features comportamentais geradas no PA005 (recência, frequência, ticket médio, diversidade, devoluções) |
| `customer_clusters` | Cluster KMeans atribuído a cada cliente + probabilidade de churn identificada no PA005 |

**Definição de churn:** cliente sem compra nos últimos 90 dias (`recency_days > 90`). Threshold de 90 dias — menor que os 120 do PA005 — para capturar clientes em risco antes do churn efetivo.

---

## Pipeline

| Etapa | Artefato | Status |
|-------|----------|--------|
| Extração de dados | `scripts/extract_raw_data.py` | ✅ |
| EDA | `notebooks/01_eda_churn.ipynb` | ✅ |
| Feature Engineering | `src/features/build_features.py` | ✅ |
| Churn Model (Camada 1) | `notebooks/02_churn_model.ipynb` | ✅ ROC-AUC 0.787 |
| A/B Testing (Camada 2) | `notebooks/03_ab_testing.ipynb` | ⏳ |
| Uplift Modeling (Camada 3) | `notebooks/04_uplift_model.ipynb` | ⏳ |

---

## Setup

```bash
pip install -r requirements.txt
```

Criar `.env` na raiz com as variáveis:

```
DB_HOST=
DB_PORT=5432
DB_NAME=ecommerce
DB_USER=
DB_PASS=
AWS_S3_BUCKET=
REFERENCE_DATE=2026-06-01
CHURN_THRESHOLD_DAYS=90
AB_TEST_SIGNIFICANCE=0.05
AB_TEST_POWER=0.80
```

---

## Estrutura de pastas

```
pa006_churn_ab_uplift/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                ← dados extraídos do RDS (não versionados)
│   ├── processed/          ← features prontas para modelagem (não versionadas)
│   └── external/
│
├── notebooks/
│   ├── 01_eda_churn.ipynb
│   ├── 02_churn_model.ipynb
│   ├── 03_ab_testing.ipynb
│   └── 04_uplift_model.ipynb
│
├── src/
│   ├── data/               ← conexão RDS, extração
│   ├── features/           ← pipeline de feature engineering
│   ├── models/             ← treino, avaliação, serialização
│   ├── experiments/        ← lógica de A/B e uplift
│   └── visualization/
│
├── models/                 ← modelos serializados (não versionados)
├── scripts/                ← scripts de execução
└── reports/
    ├── figures/
    └── exports/
```

---

## Referências

- [Causalml docs](https://causalml.readthedocs.io)
- [Statsmodels power analysis](https://www.statsmodels.org/stable/stats.html)
- CRISP-DS framework: base de todos os projetos do portfólio
