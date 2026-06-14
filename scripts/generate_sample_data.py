"""
Sample Data Generator
Generates realistic CSV datasets for testing InsightFlow AI.
Run: python scripts/generate_sample_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("sample_data")
OUT.mkdir(exist_ok=True)
np.random.seed(42)


def generate_ecommerce_sales(n=2000):
    """E-commerce sales dataset — good for revenue analysis + forecasting."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    # Add seasonality
    seasonal = np.sin(np.linspace(0, 4 * np.pi, n)) * 500 + 2000
    trend = np.linspace(1500, 3500, n)
    noise = np.random.normal(0, 300, n)
    revenue = np.abs(seasonal + trend + noise)

    df = pd.DataFrame({
        "order_date": dates,
        "order_id": ["ORD-" + str(i).zfill(6) for i in range(n)],
        "customer_id": np.random.randint(1000, 5000, n),
        "customer_name": np.random.choice(
            ["Alice Johnson", "Bob Smith", "Carol Davis", "Dave Wilson",
             "Eve Martinez", "Frank Lee", "Grace Kim", "Henry Brown"], n
        ),
        "customer_city": np.random.choice(
            ["Mumbai", "Delhi", "Pune", "Bangalore", "Chennai",
             "Hyderabad", "Kolkata", "Ahmedabad"], n
        ),
        "product": np.random.choice(
            ["Laptop Pro", "Wireless Headphones", "Smart Watch", "4K Monitor",
             "Mechanical Keyboard", "Gaming Mouse", "Webcam HD", "USB Hub"], n
        ),
        "category": np.random.choice(
            ["Computers", "Audio", "Wearables", "Displays", "Peripherals"], n
        ),
        "sales_amount": np.round(revenue / n * np.random.uniform(50, 2000, n), 2),
        "quantity": np.random.randint(1, 8, n),
        "discount_pct": np.round(np.random.uniform(0, 0.35, n), 3),
        "channel": np.random.choice(["Website", "Mobile App", "Marketplace", "Retail"], n, p=[0.5, 0.25, 0.15, 0.10]),
        "status": np.random.choice(["Delivered", "Shipped", "Processing", "Cancelled"], n, p=[0.75, 0.15, 0.07, 0.03]),
        "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.05, 0.13, 0.35, 0.45]),
    })

    # Introduce 3% missing values for realism
    for col in ["rating", "customer_city"]:
        idx = df.sample(frac=0.03).index
        df.loc[idx, col] = np.nan

    path = OUT / "ecommerce_sales.csv"
    df.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def generate_hr_dataset(n=800):
    """HR / employee dataset — good for attrition analysis."""
    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations"]
    df = pd.DataFrame({
        "employee_id": ["EMP-" + str(i).zfill(4) for i in range(n)],
        "name": ["Employee_" + str(i) for i in range(n)],
        "department": np.random.choice(departments, n),
        "role": np.random.choice(["Junior", "Mid", "Senior", "Lead", "Manager"], n),
        "salary": np.round(np.random.normal(75000, 25000, n).clip(30000, 200000), 0),
        "tenure_years": np.round(np.random.exponential(3, n).clip(0, 20), 1),
        "performance_score": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.30, 0.35, 0.20]),
        "satisfaction_score": np.round(np.random.uniform(1, 10, n), 1),
        "attrition": np.random.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "hire_date": pd.to_datetime(
            ["2024-01-01"] * n
        ) - pd.to_timedelta(np.random.randint(0, 3650, n), unit="D"),
        "city": np.random.choice(["Pune", "Bangalore", "Mumbai", "Hyderabad", "Delhi"], n),
        "remote_work": np.random.choice(["Full Remote", "Hybrid", "On-site"], n, p=[0.35, 0.40, 0.25]),
    })

    path = OUT / "hr_employees.csv"
    df.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def generate_marketing_dataset(n=3000):
    """Marketing campaign performance — good for channel analysis."""
    channels = ["Google Ads", "Facebook", "Instagram", "Email", "Organic", "Referral"]
    campaigns = ["Summer Sale", "Diwali Offer", "New Year", "Back to School", "Flash Sale"]

    df = pd.DataFrame({
        "date": pd.to_datetime(
            np.random.choice(pd.date_range("2023-01-01", "2024-12-31"), n)
        ),
        "campaign": np.random.choice(campaigns, n),
        "channel": np.random.choice(channels, n, p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]),
        "impressions": np.random.randint(1000, 500000, n),
        "clicks": np.random.randint(50, 20000, n),
        "conversions": np.random.randint(0, 500, n),
        "spend": np.round(np.random.uniform(100, 50000, n), 2),
        "revenue": np.round(np.random.uniform(200, 150000, n), 2),
        "country": np.random.choice(["India", "USA", "UK", "Germany", "Australia"], n),
    })

    df["ctr"] = np.round(df["clicks"] / df["impressions"], 4)
    df["cpc"] = np.round(df["spend"] / df["clicks"].clip(1), 2)
    df["roas"] = np.round(df["revenue"] / df["spend"].clip(0.01), 2)

    path = OUT / "marketing_campaigns.csv"
    df.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    print("Generating InsightFlow AI sample datasets...\n")
    generate_ecommerce_sales()
    generate_hr_dataset()
    generate_marketing_dataset()
    print(f"\nAll datasets saved to: {OUT.absolute()}/")
    print("\nUpload any of these to InsightFlow AI to get started!")


def generate_sqlite_db():
    """Multi-table SQLite database — tests Phase 1 SQLite ingestion."""
    import sqlite3

    db_path = OUT / "sample_store.sqlite"
    conn = sqlite3.connect(db_path)

    np.random.seed(7)
    n = 500

    # Table 1: orders
    orders = pd.DataFrame({
        "order_id":   ["ORD-" + str(i).zfill(5) for i in range(n)],
        "customer_id": np.random.randint(100, 300, n),
        "order_date":  pd.date_range("2023-01-01", periods=n, freq="12h").astype(str),
        "total_amount": np.round(np.random.uniform(20, 2000, n), 2),
        "status":      np.random.choice(["Delivered","Pending","Cancelled"], n, p=[0.75,0.15,0.10]),
        "channel":     np.random.choice(["Web","App","Store"], n),
    })

    # Table 2: customers
    customers = pd.DataFrame({
        "customer_id": range(100, 300),
        "name":       ["Customer_" + str(i) for i in range(200)],
        "city":       np.random.choice(["Pune","Mumbai","Delhi","Bangalore"], 200),
        "tier":       np.random.choice(["Bronze","Silver","Gold","Platinum"], 200),
        "signup_date": pd.date_range("2021-01-01", periods=200, freq="W").astype(str),
    })

    # Table 3: products
    products = pd.DataFrame({
        "product_id":  range(1, 51),
        "name":        ["Product_" + str(i) for i in range(1, 51)],
        "category":    np.random.choice(["Electronics","Fashion","Home","Sports"], 50),
        "price":       np.round(np.random.uniform(10, 500, 50), 2),
        "stock":       np.random.randint(0, 1000, 50),
    })

    orders.to_sql("orders", conn, if_exists="replace", index=False)
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Generated: {db_path} (3 tables: orders, customers, products)")
    return db_path


if __name__ == "__main__":
    print("Generating InsightFlow AI sample datasets...\n")
    generate_ecommerce_sales()
    generate_hr_dataset()
    generate_marketing_dataset()
    generate_sqlite_db()
    print(f"\n✅ All datasets saved to: {OUT.absolute()}/")
    print("\nUpload any of these to InsightFlow AI to get started!")
