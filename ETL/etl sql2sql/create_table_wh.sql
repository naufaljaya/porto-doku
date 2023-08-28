CREATE SCHEMA sales_wh;

USE sales_wh;
-- Create Dim_Products Table
CREATE TABLE Dim_Products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(50),
    category_id INT,
    category_name VARCHAR(50),
    price DECIMAL(10, 2),
    stock_quantity INT
);

-- Create Dim_Customers Table
CREATE TABLE Dim_Customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(50),
    phone VARCHAR(50),
    address VARCHAR(255)
);

-- Create Dim_Dates Table
CREATE TABLE Dim_Dates (
    date_id INT PRIMARY KEY,
    date DATE,
    day INT,
    month INT,
    year INT
);

-- Create Dim_Payment_Methods Table
CREATE TABLE Dim_Payment_Methods (
    payment_method_id INT PRIMARY KEY,
    payment_method ENUM("Cash", "Credit", "E-wallet")
);

-- Create Fact_Orders Table
CREATE TABLE Fact_Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    date_id INT,
    payment_method_id INT,
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES Dim_Customers(customer_id),
    FOREIGN KEY (date_id) REFERENCES Dim_Dates(date_id),
    FOREIGN KEY (payment_method_id) REFERENCES Dim_Payment_Methods(payment_method_id)
);

-- Create Fact_Order_Items Table
CREATE TABLE Fact_Order_Items (
    order_item_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    subtotal DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES Fact_Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Dim_Products(product_id)
);

-- Create Fact_Reviews Table
CREATE TABLE Fact_Reviews (
    review_id INT PRIMARY KEY,
    order_item_id INT,
    customer_id INT,
    date_id INT,
    rating INT,
    review_text TEXT,
    FOREIGN KEY (order_item_id) REFERENCES Fact_Order_Items(order_item_id),
    FOREIGN KEY (customer_id) REFERENCES Dim_Customers(customer_id),
    FOREIGN KEY (date_id) REFERENCES Dim_Dates(date_id)
);
