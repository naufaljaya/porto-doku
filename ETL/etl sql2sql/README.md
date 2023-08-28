# SQL To SQL

## Overview

- `create_table_transact.sql` - an SQL script that creates tables for a transactional database focused on sales data. The schema includes tables for categories, products, customers, orders, order items, and reviews
- `create_table_wh.sql` - an SQL script that defines a data warehouse schema using a star schema design. The schema consists of dimension tables for products, customers, dates, and payment methods, along with fact tables for orders, order items, and reviews.
- `etl.py` - demonstrates the ETL process using Python and pandas. It connects to both the transactional database and the data warehouse, extracts data from the transactional database tables, applies necessary transformations to match the data warehouse schema, and loads the transformed data into the data warehouse tables.

## Schema

### Transactional

![enter image description here](https://github.com/naufaljaya/porto-showwcase-elite-junior-data-analyst/blob/main/assets/sales_transact.PNG?raw=true)

### Data Warehouse

![enter image description here](https://github.com/naufaljaya/porto-showwcase-elite-junior-data-analyst/blob/main/assets/sales_wh.PNG?raw=true)
