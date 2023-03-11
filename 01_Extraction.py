# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 06:34:46 2023

1) Test connection to the DB
2) Determine which table to use
3) Extract and pull data offline

@author: wesch
"""
# %%
import os
# Change directory
base_path = "C:/Users/wesch/Documents/FoodRazor/"
os.chdir(base_path)


import pandas as pd
import config_fr
from sqlalchemy import create_engine
from sqlalchemy import engine

# create connection
url_object = engine.URL.create(
    "postgresql",
    username = config_fr.USER,
    password = config_fr.PASSWORD,
    host = config_fr.PGHOST,
    port = 5432,
    database= config_fr.PGDATABASE
    )

cnx = create_engine(url_object)

# %%
## Get a list of orders
tables = pd.read_sql_query("""
                           SELECT * FROM information_schema.tables
                           """, cnx)


## See what is inside the invoice_entity table
invoice_entity = pd.read_sql_query("""
                                   SELECT *
                                   FROM invoice_entity
                                   WHERE "invoiceDate" > '2022-01-01'
                                   LIMIT 10
                                   """, cnx)
                                   
product_entity = pd.read_sql_query("""
                                   SELECT *
                                   FROM invoice_product_entity
                                   
                                   LIMIT 10
                                   """, cnx)                        

## Get SG orders
all_sg_orders = pd.read_sql_query("""
WITH sg_biz AS(
SELECT id
FROM organization_entity
WHERE country = 'SG'
AND id <> 'xluid0xuk6u1fsR3pPJi'
),

biz_name AS(
SELECT DISTINCT l."businessName", l.id AS locationID
FROM location_entity AS l
INNER JOIN sg_biz AS b
ON b.id = l."organizationId"
), 

tomato_sales AS(
SELECT name AS productName, CAST(amount AS FLOAT)/10000 AS pre_tax_amount, quantity, "invoiceId", b."businessName", "organizationProductId", "organizationId"
FROM invoice_product_entity
INNER JOIN biz_name AS b 
ON b.locationID = "locationId"
WHERE "isDeleted" = 'FALSE'
)

SELECT t.productName, t.pre_tax_amount, t.quantity, t."invoiceId", i."supplierName", t."businessName", t."organizationId", i."invoiceDate"  
FROM invoice_entity AS i
INNER JOIN tomato_sales AS t 
ON i.id = t."invoiceId"
WHERE i."invoiceDate" > '2021-01-01'
ORDER BY t."organizationId", t.productName, i."invoiceDate";
""", cnx)


# %%
all_sg_orders.to_csv('./prises/all_sg_orders.csv', index=False)
