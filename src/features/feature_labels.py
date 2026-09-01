# -*- coding: utf-8 -*-
"""Traducao dos nomes tecnicos das features para rotulos em portugues, para uso no
dashboard e nos artefatos de explicabilidade (SHAP).

Usado por scripts/score_campaign.py (geracao dos parquets) e por app.py (exibicao).
"""

FEATURE_LABELS = {
    "frequency": "Número de compras",
    "avg_ticket": "Valor médio por compra",
    "gross_revenue": "Receita total",
    "revenue_velocity": "Ritmo de receita",
    "total_items": "Total de itens comprados",
    "items_velocity": "Ritmo de itens comprados",
    "basket_size": "Tamanho médio do carrinho",
    "unique_products": "Variedade de produtos",
    "product_loyalty": "Concentração em poucos produtos",
    "avg_recency_days": "Intervalo médio entre compras",
    "return_value": "Valor devolvido",
    "return_orders": "Pedidos com devolução",
    "return_rate": "Taxa de devolução",
    "return_value_ratio": "Proporção de valor devolvido",
    "customer_lifetime_days": "Tempo como cliente",
    "log_gross_revenue": "Receita total (escala log)",
    "log_total_items": "Total de itens (escala log)",
    "log_revenue_velocity": "Ritmo de receita (escala log)",
    "log_return_value": "Valor devolvido (escala log)",
    "log_basket_size": "Tamanho do carrinho (escala log)",
    "log_items_velocity": "Ritmo de itens (escala log)",
}

# unidade exibida junto ao valor; features aqui sao sempre formatadas como inteiro
_UNITS = {
    "customer_lifetime_days": "dias",
    "avg_recency_days": "dias",
    "frequency": "compras",
    "return_orders": "pedidos",
    "unique_products": "produtos",
    "total_items": "itens",
}
_MONEY = {"avg_ticket", "gross_revenue", "return_value"}
_PERCENT = {"return_rate", "return_value_ratio"}


def label_for(feature: str) -> str:
    """Rotulo em portugues da feature (fallback: o proprio nome)."""
    return FEATURE_LABELS.get(feature, feature)


def format_value(feature: str, value: float) -> str:
    """Formata o valor real da feature para um cliente, com unidade quando faz sentido."""
    if feature in _MONEY:
        return f"R$ {value:,.0f}"
    if feature in _PERCENT:
        return f"{value:.0%}"
    if feature.startswith("log_"):
        return f"{value:.1f}"
    unit = _UNITS.get(feature)
    if unit:
        return f"{value:,.0f} {unit}"
    return f"{value:,.0f}" if abs(value) >= 10 else f"{value:.1f}"
