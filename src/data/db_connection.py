"""Database connection and data extraction utilities for PA006."""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    """Create SQLAlchemy engine from environment variables.

    Returns:
        sqlalchemy.Engine: Configured database engine.
    """
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}"
        f"/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_customer_segments(engine) -> pd.DataFrame:
    """Load customer behavioral features and cluster labels from PA005.

    Source: vw_customer_segments (view over rankers + cluster metadata).

    Returns:
        DataFrame with 4,299 customers, 17 behavioral features,
        cluster_id, cluster_label, and cluster_order.
    """
    query = """
        SELECT
            customer_id,
            cluster_id,
            cluster_label,
            cluster_order,
            recency_days,
            frequency,
            avg_ticket,
            gross_revenue,
            net_revenue,
            revenue_velocity,
            total_items,
            items_velocity,
            basket_size,
            unique_products,
            product_loyalty,
            avg_recency_days,
            return_value,
            return_orders,
            return_rate,
            return_value_ratio,
            customer_lifetime_days
        FROM vw_customer_segments
        ORDER BY customer_id
    """
    return pd.read_sql(text(query), engine)


def load_rankers(engine) -> pd.DataFrame:
    """Load raw rankers table from PA005.

    Returns:
        DataFrame with 4,299 customers and numeric features + cluster.
    """
    query = "SELECT * FROM rankers ORDER BY customer_id"
    return pd.read_sql(text(query), engine)
