
#  No-SQL To SQL
This repository contains a Python script for performing an ETL (Extract, Transform, Load) process to transform data from a MongoDB database into a structured MySQL data warehouse. The ETL process is carried out using the pandas library for data manipulation and the sqlalchemy library for database interactions.

## Schema

### MongoDB Sample Record
```json
{
  "_id": {
    "$oid": "5bd761dcae323e45a93ccfe8"
  },
  "saleDate": {
    "$date": {
      "$numberLong": "1427144809506"
    }
  },
  "items": [
    {
      "name": "notepad",
      "tags": [
        "office",
        "writing",
        "school"
      ],
      "price": {
        "$numberDecimal": "35.29"
      },
      "quantity": {
        "$numberInt": "2"
      }
    },
    {
      "name": "pens",
      "tags": [
        "writing",
        "office",
        "school",
        "stationary"
      ],
      "price": {
        "$numberDecimal": "56.12"
      },
      "quantity": {
        "$numberInt": "5"
      }
    },
    {
      "name": "envelopes",
      "tags": [
        "stationary",
        "office",
        "general"
      ],
      "price": {
        "$numberDecimal": "19.95"
      },
      "quantity": {
        "$numberInt": "8"
      }
    },
    {
      "name": "binder",
      "tags": [
        "school",
        "general",
        "organization"
      ],
      "price": {
        "$numberDecimal": "14.16"
      },
      "quantity": {
        "$numberInt": "3"
      }
    }
  ],
  "storeLocation": "Denver",
  "customer": {
    "gender": "M",
    "age": {
      "$numberInt": "42"
    },
    "email": "cauho@witwuta.sv",
    "satisfaction": {
      "$numberInt": "4"
    }
  },
  "couponUsed": true,
  "purchaseMethod": "Online"
}
```


### Data Warehouse
![enter image description here](https://github.com/naufaljaya/porto-showwcase-elite-junior-data-analyst/blob/main/assets/mongodb_wh.PNG?raw=true)