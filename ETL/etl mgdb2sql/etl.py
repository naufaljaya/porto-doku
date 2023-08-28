import pymongo
import pandas as pd
from sqlalchemy import create_engine

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["your_mongodb_database"]
collection = db["your_collection"]

# Extract data from MongoDB
data_from_mongodb = collection.find({})

# Transform data with pandas
data_list = list(data_from_mongodb)
df = pd.json_normalize(data_list, record_path="items", meta=["saleDate", "storeLocation", "customer", "couponUsed", "purchaseMethod"])

# Create the Sales Fact DataFrame
sales_fact = df[["saleDate", "storeLocation", "customer.gender", "customer.age", "customer.email", "customer.satisfaction", "couponUsed", "purchaseMethod", "name", "price", "quantity"]]
sales_fact = sales_fact.rename(columns={"name": "item_name", "price": "item_price", "quantity": "item_quantity"})

# Set up MySQL connection
engine = create_engine("mysql://username:password@localhost:3306/DataWarehouse")

# Load dimension tables
item_dimension = df[["name", "price", "quantity"]].drop_duplicates().reset_index(drop=True)
tag_dimension = df["tags"].explode().drop_duplicates().reset_index(drop=True)
location_dimension = df[["storeLocation"]].drop_duplicates().reset_index(drop=True)
customer_dimension = df[["customer.gender", "customer.age", "customer.email", "customer.satisfaction"]].drop_duplicates().reset_index(drop=True)
coupon_dimension = df[["couponUsed"]].drop_duplicates().reset_index(drop=True)

item_dimension.to_sql("ItemDimension", con=engine, if_exists="replace", index=False)
tag_dimension.to_sql("TagDimension", con=engine, if_exists="replace", index=False)
location_dimension.to_sql("LocationDimension", con=engine, if_exists="replace", index=False)
customer_dimension.to_sql("CustomerDimension", con=engine, if_exists="replace", index=False)
coupon_dimension.to_sql("CouponDimension", con=engine, if_exists="replace", index=False)

# Load the Sales Fact table
sales_fact.to_sql("SalesFact", con=engine, if_exists="replace", index=False)
