import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Retention Science · PA006",
    page_icon="📊",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
THRESHOLD = 0.20
TICKET = 350
DISCOUNT = 0.10
COST_PER_CONTACT = 15
NET_REVENUE_PER_RETAINED = TICKET * (1 - DISCOUNT)  # R$315

PROFILE_COLORS = {
    "Persuadible":  "#2ecc71",
    "Sure Thing":   "#3498db",
    "Sleeping Dog": "#f39c12",
    "Lost Cause":   "#e74c3c",
}

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    churn = pd.read_parquet("data/processed/churn_scores.parquet")
    ab    = pd.read_parquet("data/processed/ab_pool_assignment.parquet")
    score = pd.read_parquet("data/processed/scoring_output.parquet")
    return churn, ab, score

churn_df, ab_df, score_df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Retention Science")
    st.caption("Churn · A/B Testing · Uplift")
    st.divider()
    page = st.radio(
        "Navegação",
        ["O Problema", "A Evidência", "Os Alvos"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("PA006 · E-commerce B2C · 4.300 clientes")

# ── Page 1: O Problema ─────────────────────────────────────────────────────────
if page == "O Problema":
    st.title("O Problema")
    st.markdown(
        "Um e-commerce B2C com 4.300 clientes perde receita silenciosamente. "
        "Clientes sem compra há **90 dias** raramente retornam espontaneamente — "
        "mas contactar toda a base em risco esgota o budget e dilui o impacto."
    )

    pool = churn_df[churn_df.churn_score >= THRESHOLD]
    churned_n = int(churn_df.churn_real.sum())
    total_n = len(churn_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes modelados", f"{total_n:,}")
    c2.metric("Churn rate histórico", f"{churned_n / total_n:.1%}")
    c3.metric("ROC-AUC (teste)", "0.776")
    c4.metric("Pool campanha (≥ 0.20)", f"{len(pool):,}")

    st.divider()

    # Score distribution
    fig = go.Figure()
    for label, color, name in [(0, "#3498db", "Ativo"), (1, "#e74c3c", "Churned")]:
        sub = churn_df[churn_df.churn_real == label]
        fig.add_trace(go.Histogram(
            x=sub.churn_score,
            name=name,
            marker_color=color,
            opacity=0.70,
            nbinsx=40,
        ))
    fig.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold {THRESHOLD}",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title="Distribuição do Score P(churn)",
        xaxis_title="P(churn)",
        yaxis_title="Clientes",
        legend_title="Status",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Por que threshold 0.20?** Falso negativo custa R$350 (receita perdida), "
        "falso positivo custa R$15 (campanha inútil). Relação FN/FP = **7×** → "
        "threshold conservador maximiza recall sem explodir o budget."
    )

    # Threshold sensitivity table
    st.subheader("Sensibilidade ao Threshold")
    rows = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        sub = churn_df[churn_df.churn_score >= t]
        tp = int((sub.churn_real == 1).sum())
        rows.append({
            "Threshold": f"{t:.2f}",
            "Pool": f"{len(sub):,}",
            "Custo campanha": f"R$ {len(sub) * COST_PER_CONTACT:,.0f}",
            "Recall": f"{tp / churned_n:.1%}",
            "Precision": f"{tp / len(sub):.1%}" if len(sub) > 0 else "—",
        })
    tbl = pd.DataFrame(rows).set_index("Threshold")
    st.dataframe(tbl, use_container_width=True)

# ── Page 2: A Evidência ────────────────────────────────────────────────────────
elif page == "A Evidência":
    st.title("A Evidência")
    st.markdown(
        "A campanha (cupom 10% + e-mail) foi testada em **1.134 clientes** do pool "
        "de risco, divididos aleatoriamente 50/50."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("χ² / p-value", "6.96 / 0.008", delta="significativo", delta_color="normal")
    c2.metric("ARR (lift absoluto)", "7.6 p.p.", delta="IC95% [2.1%, 13.1%]", delta_color="normal")
    c3.metric("Lift relativo de churn", "−20.2%")
    c4.metric("ROI da campanha", "51.7%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        ctrl_rate  = 0.376
        treat_rate = 0.300
        fig = go.Figure(data=[go.Bar(
            x=["Controle", "Tratamento"],
            y=[ctrl_rate, treat_rate],
            text=[f"{ctrl_rate:.1%}", f"{treat_rate:.1%}"],
            textposition="outside",
            marker_color=["#95a5a6", "#2ecc71"],
            width=0.45,
        )])
        fig.add_annotation(
            x=0.5, y=(ctrl_rate + treat_rate) / 2,
            xref="x", yref="y",
            text=f"↓ {ctrl_rate - treat_rate:.1%}",
            showarrow=False,
            font=dict(size=14, color="#c0392b"),
        )
        fig.update_layout(
            title="Churn Rate por Grupo",
            yaxis_title="Churn rate",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 0.50],
            template="plotly_white",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        n_treatment = 567
        custo   = n_treatment * COST_PER_CONTACT           # 8 505
        receita = n_treatment * (ctrl_rate - treat_rate) * 300  # 567 × 7.6% × R$300
        lucro   = receita - custo

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Custo campanha", "Receita recuperada", "Lucro líquido"],
            y=[-custo, receita, 0],
            text=[f"–R$ {custo:,.0f}", f"+R$ {receita:,.0f}", f"R$ {lucro:,.0f}"],
            textposition="outside",
            connector={"line": {"color": "#bdc3c7"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}},
        ))
        fig.update_layout(
            title="Impacto Financeiro por Ciclo",
            yaxis_title="R$",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"**Resultado:** campanha reduz churn de 37.6% → 30.0% (−20.2% relativo). "
        f"Lucro estimado de **R$ {lucro:,.0f}** por ciclo com ROI de **51.7%**. "
        f"NNT = 13.2 contatos para reter 1 cliente adicional."
    )

    # IC95% visualization
    st.subheader("Intervalo de Confiança do Lift (ARR)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0.021, 0.076, 0.131],
        y=[1, 1, 1],
        mode="markers+lines",
        marker=dict(size=[8, 14, 8], color=["#bdc3c7", "#2ecc71", "#bdc3c7"]),
        line=dict(color="#2ecc71", width=3),
        showlegend=False,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#e74c3c", annotation_text="H₀: ARR = 0")
    fig.update_layout(
        xaxis_title="ARR (redução absoluta de churn rate)",
        xaxis_tickformat=".1%",
        yaxis_visible=False,
        template="plotly_white",
        height=160,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Page 3: Os Alvos ──────────────────────────────────────────────────────────
else:
    st.title("Os Alvos")
    st.markdown(
        "A campanha funciona — mas não para todos. "
        "O T-Learner estima o **CATE individual** e segmenta os clientes em 4 perfis "
        "para que o budget vá apenas para quem realmente responde à intervenção."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        counts = score_df.perfil.value_counts().reset_index()
        counts.columns = ["perfil", "n"]
        fig = px.pie(
            counts,
            values="n",
            names="perfil",
            color="perfil",
            color_discrete_map=PROFILE_COLORS,
            title="Distribuição de Perfis",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        scatter_df = score_df.reset_index()
        fig = px.scatter(
            scatter_df,
            x="cate_score",
            y="churn_score",
            color="perfil",
            color_discrete_map=PROFILE_COLORS,
            title="CATE × P(churn) — Mapa de Perfis",
            labels={
                "cate_score":  "CATE (efeito incremental da campanha)",
                "churn_score": "P(churn)",
            },
            opacity=0.55,
            height=380,
            hover_data={"customer_id": True, "cate_score": ":.3f", "churn_score": ":.3f"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#7f8c8d",
                      annotation_text="CATE = 0")
        fig.add_hline(
            y=float(score_df.churn_score.median()),
            line_dash="dash",
            line_color="#7f8c8d",
            annotation_text="Mediana P(churn)",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Lista de Alvos — Persuadibles")
    st.markdown("Use o slider para ajustar o tamanho da campanha conforme o budget disponível.")

    avail = int((score_df.perfil == "Persuadible").sum())
    top_k = st.slider(
        "Top-K Persuadibles para campanha",
        min_value=50,
        max_value=min(500, avail),
        value=min(200, avail),
        step=50,
    )

    persuadibles = (
        score_df[score_df.perfil == "Persuadible"]
        .reset_index()
        .sort_values("cate_score", ascending=False)
        .head(top_k)
    )

    cate_mean = float(persuadibles.cate_score.mean())
    custo_p   = top_k * COST_PER_CONTACT
    receita_p = cate_mean * NET_REVENUE_PER_RETAINED * top_k
    roi_p     = (receita_p - custo_p) / custo_p * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persuadibles disponíveis", f"{avail:,}")
    c2.metric("Selecionados", f"{min(top_k, avail):,}")
    c3.metric("CATE médio", f"{cate_mean:.3f}")
    c4.metric("ROI projetado", f"{roi_p:.0f}%", help="CATE × ticket_líquido / custo_campanha")

    table_df = persuadibles[["customer_id", "churn_score", "cate_score", "perfil"]].copy()
    table_df = table_df.rename(columns={
        "customer_id": "Customer ID",
        "churn_score": "P(churn)",
        "cate_score":  "CATE",
        "perfil":      "Perfil",
    })
    table_df["P(churn)"] = table_df["P(churn)"].round(3)
    table_df["CATE"]     = table_df["CATE"].round(3)

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    csv_bytes = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Exportar CSV para CRM",
        data=csv_bytes,
        file_name=f"campaign_targets_top{top_k}.csv",
        mime="text/csv",
    )
