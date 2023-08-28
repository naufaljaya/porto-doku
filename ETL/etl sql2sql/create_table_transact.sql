-- Create a new schema called "sales"
CREATE SCHEMA sales;

-- Switch to the "sales" schema for table creation
USE sales;

-- Table: category
-- This table stores different product categories.
CREATE TABLE category(
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(50)
);

-- Table: product
-- This table contains details about products available in the store.
CREATE TABLE product(
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(50),
    category_id INT,
    price INT,
    stock_quantity INT,
    FOREIGN KEY (category_id) REFERENCES category(category_id)
);

-- Table: customers
-- This table stores customer information.
CREATE TABLE customers(
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_name VARCHAR(50),
    email VARCHAR(50),
    phone_number VARCHAR(20),
    address VARCHAR(50)
);

-- Table: orders
-- This table records order information.
CREATE TABLE orders(
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date DATETIME,
    total_amount INT,
    payment_amount INT,
    payment_method ENUM("Cash", "Credit", "E-wallet"),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Table: order_items
-- This table stores information about items within orders.
CREATE TABLE order_items(
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    subtotal INT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

-- Table: reviews
-- This table keeps track of product reviews provided by customers.
CREATE TABLE reviews(
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    order_item_id INT,
    customer_id INT,
    rating FLOAT,
    review_text VARCHAR(1000),
    review_date DATETIME,
    FOREIGN KEY (order_item_id) REFERENCES order_items(order_item_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
