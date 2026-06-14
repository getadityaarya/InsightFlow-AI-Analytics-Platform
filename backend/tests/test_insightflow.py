"""
Test Suite — InsightFlow AI
Covers ingestion, profiling, schema intelligence, SQL engine, chart engine, and forecasting.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sales_df():
    """Realistic sales dataset for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_date": dates,
        "customer_id": np.random.randint(1000, 2000, n),
        "customer_name": np.random.choice(["Alice", "Bob", "Carol", "Dave", "Eve"], n),
        "product": np.random.choice(["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"], n),
        "category": np.random.choice(["Electronics", "Accessories"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sales_amount": np.round(np.random.uniform(50, 2000, n), 2),
        "quantity": np.random.randint(1, 10, n),
        "discount": np.round(np.random.uniform(0, 0.3, n), 2),
    })


@pytest.fixture
def df_with_missing(sample_sales_df):
    """DataFrame with intentional missing values."""
    df = sample_sales_df.copy()
    df.loc[df.sample(frac=0.05).index, "sales_amount"] = np.nan
    df.loc[df.sample(frac=0.02).index, "customer_name"] = np.nan
    return df


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "revenue": [1000.0, 2000.0, 1500.0],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Ingestion Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestion:

    def test_load_csv(self, sample_sales_df, tmp_path):
        from app.services.ingestion import load_dataframe
        csv_path = tmp_path / "test.csv"
        sample_sales_df.to_csv(csv_path, index=False)
        result = load_dataframe(str(csv_path), "test.csv")
        assert "main" in result
        assert len(result["main"]) == 500
        assert "sales_amount" in result["main"].columns

    def test_load_excel(self, sample_sales_df, tmp_path):
        from app.services.ingestion import load_dataframe
        xlsx_path = tmp_path / "test.xlsx"
        sample_sales_df.to_excel(xlsx_path, index=False)
        result = load_dataframe(str(xlsx_path), "test.xlsx")
        assert len(result) >= 1

    def test_load_sqlite(self, sample_sales_df, tmp_path):
        import sqlite3
        from app.services.ingestion import load_dataframe
        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(db_path)
        sample_sales_df.to_sql("sales", conn, if_exists="replace", index=False)
        conn.close()
        result = load_dataframe(str(db_path), "test.sqlite")
        assert "sales" in result
        assert len(result["sales"]) == 500

    def test_unsupported_extension(self, tmp_path):
        from app.services.ingestion import load_dataframe, IngestionError
        f = tmp_path / "test.json"
        f.write_text("{}")
        with pytest.raises(IngestionError, match="Unsupported"):
            load_dataframe(str(f), "test.json")

    def test_empty_file_rejected(self, tmp_path):
        from app.services.ingestion import load_dataframe, IngestionError
        f = tmp_path / "empty.csv"
        f.write_text("col1,col2\n")
        with pytest.raises(IngestionError):
            load_dataframe(str(f), "empty.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Data Profiling Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProfiling:

    def test_profile_basic_shape(self, sample_sales_df):
        from app.services.ingestion import profile_dataframe
        profile = profile_dataframe(sample_sales_df)
        assert profile["rows"] == 500
        assert profile["columns"] == 9
        assert 0 <= profile["quality_score"] <= 100

    def test_profile_missing_values(self, df_with_missing):
        from app.services.ingestion import profile_dataframe
        profile = profile_dataframe(df_with_missing)
        assert profile["missing_values_pct"] > 0

    def test_profile_no_duplicates(self, sample_sales_df):
        from app.services.ingestion import profile_dataframe
        profile = profile_dataframe(sample_sales_df)
        assert profile["duplicate_rows"] == 0

    def test_profile_columns_present(self, sample_sales_df):
        from app.services.ingestion import profile_dataframe
        profile = profile_dataframe(sample_sales_df)
        col_names = [c["name"] for c in profile["column_profiles"]]
        assert "sales_amount" in col_names
        assert "order_date" in col_names

    def test_quality_score_high_for_clean_data(self, sample_sales_df):
        from app.services.ingestion import profile_dataframe
        profile = profile_dataframe(sample_sales_df)
        assert profile["quality_score"] >= 70


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Column Type Inference
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeInference:

    def test_infer_currency(self):
        from app.services.ingestion import _infer_column_type
        s = pd.Series([100.5, 200.0, 300.0], name="revenue")
        assert _infer_column_type(s) == "currency"

    def test_infer_numeric(self):
        from app.services.ingestion import _infer_column_type
        s = pd.Series([1, 2, 3], name="count_items")
        assert _infer_column_type(s) == "numeric"

    def test_infer_datetime(self):
        from app.services.ingestion import _infer_column_type
        s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"], name="order_date")
        assert _infer_column_type(s) == "datetime"

    def test_infer_category(self):
        from app.services.ingestion import _infer_column_type
        s = pd.Series(["A", "B", "A", "B", "A"] * 20, name="status")
        assert _infer_column_type(s) == "category"

    def test_infer_text(self):
        from app.services.ingestion import _infer_column_type
        s = pd.Series(["unique_" + str(i) for i in range(100)], name="description")
        assert _infer_column_type(s) == "text"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Schema Intelligence
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaIntelligence:

    def test_infer_metadata_shape(self, sample_sales_df):
        from app.services.schema_intelligence import infer_schema_metadata
        schema = infer_schema_metadata(sample_sales_df, "sales", "test-session")
        assert schema["table_name"] == "sales"
        assert schema["row_count"] == 500
        assert len(schema["columns"]) == 9

    def test_rag_text_generated(self, sample_sales_df):
        from app.services.schema_intelligence import infer_schema_metadata
        schema = infer_schema_metadata(sample_sales_df, "sales", "test-session")
        assert "sales_amount" in schema["rag_text"]
        assert "Table: sales" in schema["rag_text"]

    def test_pk_candidate_detected(self, sample_sales_df):
        from app.services.schema_intelligence import infer_schema_metadata
        schema = infer_schema_metadata(sample_sales_df, "sales", "test-session")
        # No true PK in our fixture (non-unique values) — just check structure
        col = next(c for c in schema["columns"] if c["name"] == "sales_amount")
        assert "is_nullable" in col

    def test_create_table_sql(self, sample_sales_df):
        from app.services.schema_intelligence import infer_schema_metadata, generate_create_table_sql
        schema = infer_schema_metadata(sample_sales_df, "sales", "test-session")
        sql = generate_create_table_sql(schema)
        assert "CREATE TABLE" in sql
        assert "[sales_amount]" in sql


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — SQL Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLValidation:

    def test_valid_select(self):
        from app.services.sql_engine import validate_sql
        sql = "SELECT customer, SUM(sales_amount) FROM main GROUP BY customer"
        result = validate_sql(sql)
        assert "SELECT" in result.upper()

    def test_rejects_delete(self):
        from app.services.sql_engine import validate_sql, SQLValidationError
        with pytest.raises(SQLValidationError):
            validate_sql("DELETE FROM main WHERE id = 1")

    def test_rejects_drop(self):
        from app.services.sql_engine import validate_sql, SQLValidationError
        with pytest.raises(SQLValidationError):
            validate_sql("DROP TABLE main")

    def test_rejects_update(self):
        from app.services.sql_engine import validate_sql, SQLValidationError
        with pytest.raises(SQLValidationError):
            validate_sql("UPDATE main SET col = 1")

    def test_rejects_insert(self):
        from app.services.sql_engine import validate_sql, SQLValidationError
        with pytest.raises(SQLValidationError):
            validate_sql("INSERT INTO main VALUES (1, 2)")

    def test_rejects_syntax_error(self):
        from app.services.sql_engine import validate_sql, SQLValidationError
        with pytest.raises(SQLValidationError):
            validate_sql("SELECT * FORM main")  # typo: FORM not FROM

    def test_strips_semicolon(self):
        from app.services.sql_engine import validate_sql
        result = validate_sql("SELECT 1;")
        assert not result.endswith(";")

    def test_allows_complex_select(self):
        from app.services.sql_engine import validate_sql
        sql = """
        SELECT region, product, SUM(sales_amount) AS revenue,
               COUNT(*) AS orders, AVG(discount) AS avg_discount
        FROM main
        WHERE order_date >= '2024-01-01'
        GROUP BY region, product
        HAVING SUM(sales_amount) > 1000
        ORDER BY revenue DESC
        LIMIT 20
        """
        result = validate_sql(sql)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — SQL Execution
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLExecution:

    def test_basic_select(self, sample_sales_df):
        from app.services.sql_engine import execute_sql_on_dataframes
        result = execute_sql_on_dataframes(
            "SELECT * FROM main LIMIT 10",
            {"main": sample_sales_df},
            "test-session",
        )
        assert result["row_count"] == 10
        assert "sales_amount" in result["columns"]

    def test_aggregation(self, sample_sales_df):
        from app.services.sql_engine import execute_sql_on_dataframes
        result = execute_sql_on_dataframes(
            "SELECT region, SUM(sales_amount) AS total FROM main GROUP BY region",
            {"main": sample_sales_df},
            "test-session",
        )
        assert result["row_count"] == 4  # North, South, East, West
        assert "total" in result["columns"]

    def test_caching(self, sample_sales_df):
        from app.services.sql_engine import execute_sql_on_dataframes
        sql = "SELECT COUNT(*) AS cnt FROM main"
        frames = {"main": sample_sales_df}
        r1 = execute_sql_on_dataframes(sql, frames, "test-cache-session")
        r2 = execute_sql_on_dataframes(sql, frames, "test-cache-session")
        assert r2["cached"] is True

    def test_empty_result(self, sample_sales_df):
        from app.services.sql_engine import execute_sql_on_dataframes
        result = execute_sql_on_dataframes(
            "SELECT * FROM main WHERE sales_amount < 0",
            {"main": sample_sales_df},
            "test-session",
        )
        assert result["row_count"] == 0

    def test_null_handling(self, df_with_missing):
        from app.services.sql_engine import execute_sql_on_dataframes
        result = execute_sql_on_dataframes(
            "SELECT COALESCE(sales_amount, 0) AS amount FROM main LIMIT 50",
            {"main": df_with_missing},
            "test-session",
        )
        assert result["row_count"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Chart Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestChartEngine:

    def test_auto_select_line_for_datetime(self):
        from app.services.chart_engine import select_chart_type
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30),
            "revenue": range(30),
        })
        assert select_chart_type(df) == "line"

    def test_auto_select_bar_for_category_numeric(self):
        from app.services.chart_engine import select_chart_type
        # 3 cols prevents pie rule (which only fires on exactly 2 cols)
        df = pd.DataFrame({
            "product": ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20,
            "region": ["North"] * 40 + ["South"] * 40,
            "sales": list(range(80)),
        })
        assert select_chart_type(df) == "bar"

    def test_auto_select_pie_for_two_cols(self):
        from app.services.chart_engine import select_chart_type
        # Pie needs exactly 2 cols: 1 text/category + 1 numeric
        # Must have low enough unique ratio to be "category" type
        region = ["North"] * 40 + ["South"] * 40 + ["East"] * 20
        share = [33.3] * 40 + [33.3] * 40 + [33.4] * 20
        df = pd.DataFrame({"region": region, "share": share})
        assert select_chart_type(df) == "pie"

    def test_auto_select_scatter_for_two_numerics(self):
        from app.services.chart_engine import select_chart_type
        df = pd.DataFrame({
            "price": [10.0, 20.0, 30.0],
            "quantity": [5, 3, 8],
        })
        assert select_chart_type(df) == "scatter"

    def test_build_bar_config(self):
        from app.services.chart_engine import build_plotly_config
        df = pd.DataFrame({"product": ["A", "B"], "sales": [100, 200]})
        config = build_plotly_config(df, "bar", "Test Chart")
        assert config["type"] == "bar"
        assert len(config["data"]) > 0

    def test_build_line_config(self):
        from app.services.chart_engine import build_plotly_config
        df = pd.DataFrame({
            "order_date": pd.date_range("2024-01-01", periods=10),
            "revenue": range(10),
        })
        config = build_plotly_config(df, "line", "Revenue Trend")
        assert config["type"] == "line"

    def test_empty_df_returns_empty_chart(self):
        from app.services.chart_engine import build_plotly_config
        df = pd.DataFrame()
        config = build_plotly_config(df, "bar")
        assert config["type"] == "empty"

    def test_table_fallback(self):
        from app.services.chart_engine import build_plotly_config
        df = pd.DataFrame({"col_a": ["x", "y"], "col_b": ["p", "q"]})
        config = build_plotly_config(df, "table")
        assert config["type"] == "table"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 15 — Forecasting
# ─────────────────────────────────────────────────────────────────────────────

class TestForecasting:

    def test_is_forecast_question(self):
        from app.services.forecasting import is_forecast_question
        assert is_forecast_question("forecast next 6 months sales") is True
        assert is_forecast_question("predict revenue for next quarter") is True
        assert is_forecast_question("show top products") is False
        assert is_forecast_question("what is the total revenue?") is False

    def test_parse_forecast_periods_monthly(self):
        from app.services.forecasting import parse_forecast_periods
        periods, freq = parse_forecast_periods("forecast next 3 months")
        assert freq == "M"
        assert periods == 3

    def test_parse_forecast_periods_weekly(self):
        from app.services.forecasting import parse_forecast_periods
        periods, freq = parse_forecast_periods("predict next 8 weeks")
        assert freq == "W"

    def test_parse_forecast_default(self):
        from app.services.forecasting import parse_forecast_periods
        periods, freq = parse_forecast_periods("what will sales be in the future?")
        assert freq == "D"
        assert periods == 30

    def test_detect_forecast_columns(self, sample_sales_df):
        from app.services.forecasting import detect_forecast_columns
        result = detect_forecast_columns(sample_sales_df)
        assert result is not None
        date_col, value_col = result
        assert "date" in date_col.lower() or "date" in date_col
        assert date_col in sample_sales_df.columns

    def test_forecast_not_enough_data(self):
        from app.services.forecasting import run_forecast, ForecastError
        tiny_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=5),
            "y": [1, 2, 3, 4, 5],
        })
        # Should raise ForecastError for any reason (not enough data OR prophet missing)
        with pytest.raises(ForecastError):
            run_forecast(tiny_df, "ds", "y", periods=7)

    @pytest.mark.slow
    def test_full_forecast(self, sample_sales_df):
        """Integration test — requires prophet installed. Skip with -m 'not slow'."""
        from app.services.forecasting import run_forecast
        try:
            result = run_forecast(sample_sales_df, "order_date", "sales_amount", periods=30)
            assert "summary" in result
            assert result["direction"] in ("increase", "decrease")
            assert len(result["forecast"]) == len(sample_sales_df) + 30
        except ImportError:
            pytest.skip("Prophet not installed")


# ─────────────────────────────────────────────────────────────────────────────
# Integration — end-to-end pipeline (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipeline:

    def test_csv_to_profile_to_schema(self, sample_sales_df, tmp_path):
        """Full ingestion → profile → schema pipeline."""
        from app.services.ingestion import load_dataframe, profile_dataframe, save_session_data
        from app.services.schema_intelligence import infer_schema_metadata

        # Save CSV
        csv_path = tmp_path / "sales.csv"
        sample_sales_df.to_csv(csv_path, index=False)

        # Ingest
        dfs = load_dataframe(str(csv_path), "sales.csv")
        assert "main" in dfs

        df = dfs["main"]

        # Profile
        profile = profile_dataframe(df, "main")
        assert profile["quality_score"] > 0

        # Schema
        schema = infer_schema_metadata(df, "main", "integration-test-session")
        assert schema["rag_text"]
        assert len(schema["columns"]) == len(df.columns)

    def test_sql_pipeline(self, sample_sales_df):
        """SQL generate → validate → execute pipeline."""
        from app.services.sql_engine import validate_sql, execute_sql_on_dataframes

        sql = "SELECT region, SUM(sales_amount) AS revenue FROM main GROUP BY region ORDER BY revenue DESC"

        # Validate
        validated = validate_sql(sql)
        assert validated

        # Execute
        result = execute_sql_on_dataframes(validated, {"main": sample_sales_df}, "test-e2e")
        assert result["row_count"] == 4
        assert result["columns"] == ["region", "revenue"]
