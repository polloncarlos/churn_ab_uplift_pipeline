import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(
    page_title="Painel de Retenção · CRM",
    page_icon="🎯",
    layout="wide",
)

# ── Constantes de negócio ──────────────────────────────────────────────────────
TICKET          = 350
DESCONTO        = 0.10
CUSTO_CONTATO   = 15
RECEITA_RETIDO  = TICKET * (1 - DESCONTO)   # R$315 por churner retido
CORTE_RISCO     = 0.20

CORES_PERFIL = {
    "Persuadible":  "#2ecc71",
    "Sure Thing":   "#3498db",
    "Sleeping Dog": "#f39c12",
    "Lost Cause":   "#e74c3c",
}

LABEL_PERFIL = {
    "Persuadible":  "Responde à campanha",
    "Sure Thing":   "Retorna sem contato",
    "Sleeping Dog": "Contato contraproducente",
    "Lost Cause":   "Fora do alcance",
}

# ── Dados ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    churn = pd.read_parquet("data/processed/churn_scores.parquet")
    ab    = pd.read_parquet("data/processed/ab_pool_assignment.parquet")
    score = pd.read_parquet("data/processed/scoring_output.parquet")
    score = score.reset_index()
    score["Perfil Negócio"] = score["perfil"].map(LABEL_PERFIL)
    shap_contrib = pd.read_parquet("data/processed/shap_contributions.parquet")
    shap_global  = pd.read_parquet("data/processed/shap_global.parquet")
    return churn, ab, score, shap_contrib, shap_global

churn_df, ab_df, score_df, shap_contrib_df, shap_global_df = load_data()

# Persuadíveis pré-computados para download disponível em todas as páginas
_persuadiveis_all = (
    score_df[score_df.perfil == "Persuadible"]
    .sort_values("cate_score", ascending=False)
    .copy()
)
_N_DEFAULT = min(200, len(_persuadiveis_all))

def _build_tabela(df_alvos: pd.DataFrame) -> pd.DataFrame:
    t = df_alvos[["customer_id", "churn_score", "cate_score"]].copy()
    t.columns = ["ID do Cliente", "Risco de Churn", "Potencial de Resposta"]
    t["Risco de Churn"]        = (t["Risco de Churn"] * 100).round(1).astype(str) + "%"
    t["Potencial de Resposta"] = (t["Potencial de Resposta"] * 100).round(1).astype(str) + "%"
    t["Ação Recomendada"]      = "Enviar cupom 10% + e-mail"
    return t

def calcular_roi(n: int, cate_medio: float) -> float:
    receita = cate_medio * RECEITA_RETIDO * n
    custo   = n * CUSTO_CONTATO
    return (receita - custo) / custo * 100 if custo > 0 else 0.0

def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lista Campanha")
        ws = writer.sheets["Lista Campanha"]
        fill   = PatternFill("solid", fgColor="1F4E79")
        fonte  = Font(bold=True, color="FFFFFF")
        centro = Alignment(horizontal="center")
        for cell in ws[1]:
            cell.fill      = fill
            cell.font      = fonte
            cell.alignment = centro
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 22
    return buf.getvalue()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Painel de Retenção")
    st.caption("Time de CRM · E-commerce B2C")
    st.divider()
    pagina = st.radio(
        "Navegação",
        ["Visão Geral", "Resultado do Teste", "Lista de Ação", "Como o Modelo Decide"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Campanha: cupom 10% + e-mail · R$ 15/contato")
    st.divider()
    _n_dl     = st.session_state.get("n_sel", _N_DEFAULT)
    _tab_dl   = _build_tabela(_persuadiveis_all.head(_n_dl))
    st.download_button(
        label=f"⬇ Baixar Excel ({_n_dl} clientes)",
        data=to_excel(_tab_dl),
        file_name=f"lista_campanha_{_n_dl}_clientes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.download_button(
        label="⬇ Baixar CSV",
        data=_tab_dl.to_csv(index=False).encode("utf-8"),
        file_name=f"lista_campanha_{_n_dl}_clientes.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Visão Geral":
    st.title("Visão Geral")
    st.markdown(
        "Clientes que não compram há **90 dias** raramente retornam espontaneamente. "
        "O modelo identifica quem está em risco e prioriza quem realmente responde à campanha."
    )

    em_risco      = churn_df[churn_df.churn_score >= CORTE_RISCO]
    n_risco       = len(em_risco)
    n_persuadivel = int((score_df.perfil == "Persuadible").sum())
    receita_risco = n_risco * TICKET
    custo_camp    = n_persuadivel * CUSTO_CONTATO

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes em risco", f"{n_risco:,}",
              help="Clientes com alta probabilidade de não retornar nos próximos 90 dias")
    c2.metric("Receita em risco", f"R$ {receita_risco:,.0f}",
              help="Estimativa de receita perdida caso esses clientes churnem")
    c3.metric("Custo da campanha (alvos)", f"R$ {custo_camp:,.0f}",
              help=f"{n_persuadivel} clientes que respondem à campanha × R$ {CUSTO_CONTATO}")
    c4.metric("ROI validado em teste", "51,7%",
              help="Resultado confirmado em teste A/B com 1.134 clientes")

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = go.Figure()
        for label, cor, nome in [(0, "#3498db", "Clientes ativos"), (1, "#e74c3c", "Clientes perdidos")]:
            sub = churn_df[churn_df.churn_real == label]
            fig.add_trace(go.Histogram(
                x=sub.churn_score,
                name=nome,
                marker_color=cor,
                opacity=0.70,
                nbinsx=40,
            ))
        fig.add_vline(
            x=CORTE_RISCO,
            line_dash="dash",
            line_color="#f39c12",
            annotation_text="Corte de risco",
            annotation_position="top right",
        )
        fig.update_layout(
            barmode="overlay",
            title="Distribuição de Risco — Base de Clientes",
            xaxis_title="Nível de Risco de Churn",
            yaxis_title="Número de Clientes",
            legend_title="Status",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Como funciona")
        st.markdown("""
O modelo analisa o comportamento de compra de cada cliente e atribui um **nível de risco de churn**.

**Clientes acima do corte de risco** entram no pool da campanha.

Dentro desse pool, um segundo modelo identifica **quem realmente responde** ao cupom, evitando desperdício de budget com quem voltaria sozinho.

| Perfil | O que fazer |
|--------|------------|
| Responde à campanha | ✅ Contatar (prioridade) |
| Retorna sem contato | ⏸ Monitorar sem cupom |
| Contato contraproducente | ⚠️ Evitar |
| Fora do alcance | ❌ Não contatar |
""")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — RESULTADO DO TESTE
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Resultado do Teste":
    st.title("Resultado do Teste")
    st.markdown(
        "A campanha foi testada com **1.134 clientes** divididos em dois grupos iguais. "
        "Apenas um grupo recebeu o cupom. O outro serviu de comparação."
    )

    st.success("✅ **Campanha validada.** O cupom reduziu o churn de forma consistente e o resultado não é obra do acaso.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Redução de churn", "7,6 p.p.", help="Diferença entre o grupo sem e com campanha")
    c2.metric("Redução relativa", "−20,2%", help="O grupo com cupom teve 20% menos churn que o grupo controle")
    c3.metric("ROI da campanha", "51,7%", help="Lucro líquido ÷ custo da campanha")
    c4.metric("Retorno por ciclo", "R$ 4.400", help="Lucro estimado por ciclo de campanha (567 clientes contactados)")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[go.Bar(
            x=["Sem campanha", "Com campanha"],
            y=[0.376, 0.300],
            text=["37,6%", "30,0%"],
            textposition="outside",
            marker_color=["#95a5a6", "#2ecc71"],
            width=0.45,
        )])
        fig.add_annotation(
            x=0.5, y=0.340, xref="x", yref="y",
            text="↓ 7,6 pontos percentuais",
            showarrow=False,
            font=dict(size=13, color="#c0392b"),
        )
        fig.update_layout(
            title="Taxa de Churn por Grupo",
            yaxis_title="% de clientes perdidos",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 0.50],
            template="plotly_white",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        n_trat  = 567
        custo   = n_trat * CUSTO_CONTATO
        receita = n_trat * 0.076 * 300
        lucro   = receita - custo

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Custo da campanha", "Clientes recuperados", "Lucro líquido"],
            y=[-custo, receita, 0],
            text=[f"−R$ {custo:,.0f}", f"+R$ {receita:,.0f}", f"R$ {lucro:,.0f}"],
            textposition="outside",
            connector={"line": {"color": "#bdc3c7"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#1F4E79"}},
        ))
        fig.update_layout(
            title="Impacto Financeiro por Ciclo de Campanha",
            yaxis_title="R$",
            template="plotly_white",
            height=430,
            margin=dict(t=40, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div style="background-color:#dbeafe;padding:1rem 1.25rem;border-radius:0.5rem;'
        f'border-left:4px solid #3b82f6;font-size:0.95rem;">'
        f'<b>Como interpretar:</b> De cada 13 clientes contactados, 1 é retido graças ao cupom. '
        f'Com {n_trat} contatos, a campanha gerou <b>R$ {receita:,.0f}</b> em receita recuperada '
        f'a um custo de R$ {custo:,.0f}. Lucro estimado por ciclo: <b>R$ {lucro:,.0f}</b>.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — LISTA DE AÇÃO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Lista de Ação":
    st.title("Lista de Ação")
    st.markdown(
        "Use o slider para ajustar o budget disponível. "
        "A lista seleciona automaticamente os clientes com **maior potencial de resposta à campanha**."
    )

    persuadiveis = (
        score_df[score_df.perfil == "Persuadible"]
        .sort_values("cate_score", ascending=False)
        .copy()
    )
    n_disp = len(persuadiveis)

    # ── Slider de budget ──────────────────────────────────────────────────────
    budget_max = n_disp * CUSTO_CONTATO
    budget = st.slider(
        "Budget da campanha (R$)",
        min_value=CUSTO_CONTATO * 10,
        max_value=budget_max,
        value=min(3000, budget_max),
        step=CUSTO_CONTATO * 10,
        format="R$ %d",
    )
    n_sel = min(budget // CUSTO_CONTATO, n_disp)
    st.session_state["n_sel"] = n_sel
    alvos = persuadiveis.head(n_sel)
    cate_medio = float(alvos.cate_score.mean()) if n_sel > 0 else 0.0
    roi = calcular_roi(n_sel, cate_medio)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes selecionados", f"{n_sel:,}",
              help=f"De {n_disp} disponíveis que respondem à campanha")
    c2.metric("Custo total", f"R$ {n_sel * CUSTO_CONTATO:,.0f}")
    c3.metric("Potencial de resposta médio", f"{cate_medio:.1%}",
              help="Incremento esperado na taxa de retenção para este grupo")
    c4.metric("ROI projetado", f"{roi:.0f}%")

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        contagem = score_df.perfil.value_counts().reset_index()
        contagem.columns = ["perfil", "n"]
        contagem["label"] = contagem.perfil.map(LABEL_PERFIL)
        fig = px.pie(
            contagem,
            values="n",
            names="label",
            color="perfil",
            color_discrete_map=CORES_PERFIL,
            title="Perfis no Pool de Risco",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        scatter_df = score_df.copy()
        scatter_df["label"] = scatter_df.perfil.map(LABEL_PERFIL)
        fig = px.scatter(
            scatter_df,
            x="cate_score",
            y="churn_score",
            color="label",
            color_discrete_map={v: CORES_PERFIL[k] for k, v in LABEL_PERFIL.items()},
            title="Risco de Churn × Potencial de Resposta",
            labels={
                "cate_score":  "Potencial de Resposta à Campanha",
                "churn_score": "Risco de Churn",
                "label":       "Perfil",
            },
            opacity=0.55,
            height=380,
            hover_data={
                "customer_id": True,
                "cate_score":  ":.3f",
                "churn_score": ":.3f",
                "label":       True,
            },
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#bdc3c7")
        fig.add_hline(
            y=float(score_df.churn_score.median()),
            line_dash="dash",
            line_color="#bdc3c7",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Lista de contatos — {n_sel} clientes selecionados")

    tabela = _build_tabela(alvos)

    st.dataframe(tabela, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — COMO O MODELO DECIDE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Como o Modelo Decide")
    st.markdown(
        "O modelo de risco não é uma caixa-preta. Aqui você vê **o que ele mais considera "
        "no geral** e **por que um cliente específico** entrou na lista."
    )

    # ── Visão global ─────────────────────────────────────────────────────────
    st.subheader("O que o modelo mais olha")
    g = shap_global_df.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=g.mean_abs_shap,
        y=g.feature_label,
        orientation="h",
        marker_color="#1F4E79",
    ))
    fig.update_layout(
        title="Peso médio de cada fator na decisão do modelo",
        xaxis_title="Influência média (quanto maior, mais o fator pesa)",
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Barra maior = fator que o modelo mais usa para separar quem tende a churnar de "
        "quem tende a ficar. É a visão geral; para um cliente específico o peso muda."
    )

    st.divider()

    # ── Drill-down por cliente ───────────────────────────────────────────────
    st.subheader("Por que este cliente está na lista?")

    pool_ids = (
        score_df.sort_values("churn_score", ascending=False)["customer_id"].tolist()
    )
    cid = st.selectbox(
        "Cliente do pool de risco",
        pool_ids,
        format_func=lambda x: f"Cliente {x}",
    )

    cinfo = score_df[score_df.customer_id == cid].iloc[0]
    m1, m2 = st.columns(2)
    m1.metric("Risco de churn", f"{cinfo.churn_score:.0%}")
    m2.metric("Perfil", LABEL_PERFIL.get(cinfo.perfil, cinfo.perfil))

    c = shap_contrib_df[shap_contrib_df.customer_id == cid].copy()

    top_risco = c[c.direcao == "aumenta risco"].nlargest(2, "shap_value")
    if len(top_risco):
        motivos = " e ".join(
            f"{r.feature_label.lower()} ({r.feature_value_fmt})"
            for r in top_risco.itertuples()
        )
        st.markdown(f"**Os fatores que mais pesaram para incluir este cliente:** {motivos}.")

    c = c.iloc[c.shap_value.abs().argsort()]  # menor peso embaixo, maior no topo
    labels = [f"{r.feature_label} ({r.feature_value_fmt})" for r in c.itertuples()]
    cores = ["#e74c3c" if v > 0 else "#2ecc71" for v in c.shap_value]

    fig = go.Figure(go.Bar(
        x=c.shap_value,
        y=labels,
        orientation="h",
        marker_color=cores,
        text=[f"{v:+.2f}" for v in c.shap_value],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#7f8c8d")
    fig.update_layout(
        title="Fatores que empurraram este cliente para dentro (vermelho) ou para fora (verde) da lista",
        xaxis_title="←  reduz risco        |        aumenta risco  →",
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=50, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada barra é o peso daquele fator na decisão do modelo para este cliente. "
        "O valor entre parênteses é o dado real do cliente."
    )
