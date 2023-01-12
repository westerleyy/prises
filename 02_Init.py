# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 06:34:46 2023

1) Explore orders from SG between 2020 and now

@author: wesch
"""
# %%

import pandas as pd
import os

# Change directory
cwd = os.getcwd()
os.chdir(cwd + "\prises")

# Import data
sg_orders = pd.read_csv('all_sg_orders.csv')

# %%
# REFERENCE POINT
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
SELECT name AS productName, CAST(amount AS FLOAT)/10000 AS amount, "createdAt", "invoiceId", b."businessName", "organizationProductId", "organizationId"
FROM invoice_product_entity
INNER JOIN biz_name AS b 
ON b.locationID = "locationId"
WHERE "isDeleted" = 'FALSE'
)

SELECT t.productName, t.amount, t."createdAt", t."invoiceId", i."supplierName", t."businessName", t."organizationId", t."organizationProductId", i."invoiceDate"  
FROM invoice_entity AS i
INNER JOIN tomato_sales AS t 
ON i.id = t."invoiceId"
WHERE i."invoiceDate" > '2021-01-01'
ORDER BY t."organizationId", t.productName, t."createdAt";
""", cnx)


# %%

