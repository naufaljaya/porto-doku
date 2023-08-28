-- Create the database
CREATE SCHEMA DataWarehouse;

-- Use the database
USE DataWarehouse;

-- Create the Time Dimension table
CREATE TABLE TimeDimension (
    time_id INT PRIMARY KEY AUTO_INCREMENT,
    saleDate DATE
);

-- Create the Item Dimension table
CREATE TABLE ItemDimension (
    item_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    price DECIMAL(10, 2),
    quantity INT
);

-- Create the Tag Dimension table
CREATE TABLE TagDimension (
    tag_id INT PRIMARY KEY AUTO_INCREMENT,
    tag_name VARCHAR(255)
);

-- Create the ItemTag Junction table
CREATE TABLE ItemTag (
    item_id INT,
    tag_id INT,
    PRIMARY KEY (item_id, tag_id),
    FOREIGN KEY (item_id) REFERENCES ItemDimension(item_id),
    FOREIGN KEY (tag_id) REFERENCES TagDimension(tag_id)
);

-- Create the Location Dimension table
CREATE TABLE LocationDimension (
    location_id INT PRIMARY KEY AUTO_INCREMENT,
    storeLocation VARCHAR(255)
);

-- Create the Customer Dimension table
CREATE TABLE CustomerDimension (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    gender CHAR(1),
    age INT,
    email VARCHAR(255),
    satisfaction INT
);

-- Create the Coupon Dimension table
CREATE TABLE CouponDimension (
    coupon_id INT PRIMARY KEY AUTO_INCREMENT,
    couponUsed BOOLEAN
);

-- Create the Sales Fact table
CREATE TABLE SalesFact (
    sale_id INT PRIMARY KEY AUTO_INCREMENT,
    time_id INT,
    item_id INT,
    location_id INT,
    customer_id INT,
    coupon_id INT,
    purchaseMethod VARCHAR(255),
    FOREIGN KEY (time_id) REFERENCES TimeDimension(time_id),
    FOREIGN KEY (item_id) REFERENCES ItemDimension(item_id),
    FOREIGN KEY (location_id) REFERENCES LocationDimension(location_id),
    FOREIGN KEY (customer_id) REFERENCES CustomerDimension(customer_id),
    FOREIGN KEY (coupon_id) REFERENCES CouponDimension(coupon_id)
);
