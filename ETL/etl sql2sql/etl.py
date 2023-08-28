import pandas as pd
from sqlalchemy import create_engine

# Connect to your transactional database
transactional_db_url = "your_transactional_db_connection_string"
engine_transactional = create_engine(transactional_db_url)

# Connect to your data warehouse database
warehouse_db_url = "your_data_warehouse_connection_string"
engine_warehouse = create_engine(warehouse_db_url)

# Load data from the transactional database
categories_df = pd.read_sql_table("category", engine_transactional)
products_df = pd.read_sql_table("product", engine_transactional)
customers_df = pd.read_sql_table("customers", engine_transactional)
orders_df = pd.read_sql_table("orders", engine_transactional)
order_items_df = pd.read_sql_table("order_items", engine_transactional)
reviews_df = pd.read_sql_table("reviews", engine_transactional)

# Transform data to fit the data warehouse schema

# Dim_Products
dim_products_df = products_df.drop(columns=["category_name"])
dim_products_df = dim_products_df.rename(columns={"price": "price", "stock_quantity": "stock_quantity"})

# Dim_Customers - No transformation needed

# Dim_Dates
dim_dates_df = orders_df[["order_date"]].copy()
dim_dates_df["date"] = dim_dates_df["order_date"].dt.date
dim_dates_df["day"] = dim_dates_df["order_date"].dt.day
dim_dates_df["month"] = dim_dates_df["order_date"].dt.month
dim_dates_df["year"] = dim_dates_df["order_date"].dt.year

# Dim_Payment_Methods
dim_payment_methods_df = pd.DataFrame({"payment_method": ["Cash", "Credit", "E-wallet"]})

# Fact_Orders
fact_orders_df = orders_df.drop(columns=["order_date", "payment_amount"])
fact_orders_df = fact_orders_df.rename(columns={"total_amount": "total_amount"})

# Fact_Order_Items - No transformation needed

# Fact_Reviews
fact_reviews_df = reviews_df.drop(columns=["rating", "review_text"])
fact_reviews_df = fact_reviews_df.rename(columns={"order_item_id": "order_item_id", "customer_id": "customer_id"})

# Load transformed data into the data warehouse
dim_products_df.to_sql("Dim_Products", con=engine_warehouse, index=False, if_exists="replace")
customers_df.to_sql("Dim_Customers", con=engine_warehouse, index=False, if_exists="replace")
dim_dates_df.to_sql("Dim_Dates", con=engine_warehouse, index=False, if_exists="replace")
dim_payment_methods_df.to_sql("Dim_Payment_Methods", con=engine_warehouse, index=False, if_exists="replace")
fact_orders_df.to_sql("Fact_Orders", con=engine_warehouse, index=False, if_exists="replace")
order_items_df.to_sql("Fact_Order_Items", con=engine_warehouse, index=False, if_exists="replace")
fact_reviews_df.to_sql("Fact_Reviews", con=engine_warehouse, index=False, if_exists="replace")
