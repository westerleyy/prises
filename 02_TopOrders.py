# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 06:34:46 2023

1) Explore orders from SG between 2021 and now
2) Find the top twenty from each organization to look for keywords

@author: wesch
"""

# %%
## Download tiny roberta
## Once is enough

###from huggingface_hub import hf_hub_download
###hf_hub_download(repo_id="deepset/tinyroberta-squad2", filename="config.json", cache_dir= r"C:\Users\wesch\Documents\FoodRazor\prises\tiny_roberta")
# %%
import pandas as pd
import os
import seaborn as sns

# Change directory
cwd = os.getcwd()
os.chdir(cwd + "\prises")

# Import data
# Perform basic data type conversion
sg_orders = pd.read_csv('all_sg_orders.csv')
sg_orders['total_pre_tax_amount'] = sg_orders['pre_tax_amount'] * sg_orders['quantity']
sg_orders['invoiceDate'] = pd.to_datetime(sg_orders['invoiceDate'], errors = 'coerce')
sg_orders['invoiceYear'] = pd.DatetimeIndex(sg_orders['invoiceDate']).year
sg_orders['invoiceMonth'] = pd.DatetimeIndex(sg_orders['invoiceDate']).month

## Remove anomalies and February 2023 orders
sg_orders = sg_orders.loc[sg_orders['invoiceYear']<=2023].copy()
sg_orders = sg_orders.loc[sg_orders['invoiceDate']<'2023-2-1'].copy()

# Sample
sg_orders_sample = sg_orders.sample(n = 1000)

product_name_list = sg_orders_sample.productname.to_list()

# %%
## Clean product names first
sg_orders['cl_productname'] = sg_orders['productname'].str.replace('\w*\d\w*', '', regex = True)
sg_orders['cl_productname'] = sg_orders['cl_productname'].str.replace('[\(\[].*?[\)\]]', '', regex = True)
sg_orders['cl_productname'] = sg_orders['cl_productname'].str.replace('[^0-9a-zA-z ]+', '', regex = True)

common_units = [' X ', 'KG', 'KGS', 'CTN', 'FFB', 'PKT', ' PC', 'PCS', ' GR ', 'QTY']

for u in common_units:
    sg_orders['cl_productname'] = sg_orders['cl_productname'].str.replace(u, '')

sg_orders['cl_productname'] = sg_orders['cl_productname'].str.rstrip()
sg_orders['cl_productname'] = sg_orders['cl_productname'].str.lstrip()
sg_orders['cl_productname'] = sg_orders['cl_productname'].apply(lambda x: " ".join(n for n in x.split(' ') if len(n)>1))

### Drop those with empty string
sg_orders_clean = sg_orders.loc[sg_orders['cl_productname'] != ''].copy()


## Look for 20 most ordered items per organization
## Sum to account for products that now have the same name after cleaning up
count_per_org = sg_orders_clean[['cl_productname', 'organizationId', 'invoiceId']].groupby(['cl_productname', 'organizationId']).count().reset_index()
count_per_org.rename(columns = {'invoiceId': 'order_frequency'}, inplace = True)
count_per_org = count_per_org.groupby(['cl_productname', 'organizationId']).sum().reset_index()
count_per_org.sort_values(ascending = False, by = ['order_frequency'], inplace = True)
count_per_org.reset_index(inplace = True, drop = True)

# Get a list of organizations
org_list = list(count_per_org['organizationId'].unique())

## Return top orders
top_orders_df = pd.DataFrame()
for i in org_list:
    organization_orders = count_per_org.loc[count_per_org['organizationId'] == i].copy().reset_index(drop = True)
    organization_orders = organization_orders.iloc[0:20,].copy()
    top_orders_df = pd.concat([organization_orders, top_orders_df])

# %%
## Count the number of items that were ordered more than once, grouped by organization and supplier
## 96k items were ordered more than once by each organization
sg_orders_count = sg_orders[['productname', 'supplierName', 'organizationId', 'invoiceId']].groupby(['productname', 'supplierName', 'organizationId']).count().reset_index()
sg_orders_count_repeat = sg_orders_count.loc[sg_orders_count['invoiceId']>1].copy()

## Count the number of items that were ordered more than once, grouped by supplier
## 76k items were ordered more than once by organizations from the same supplier in general
sg_orders_supplier_count = sg_orders[['productname', 'supplierName', 'invoiceId']].groupby(['productname', 'supplierName']).count().reset_index()
sg_orders_supplier_count_repeat = sg_orders_supplier_count.loc[sg_orders_supplier_count['invoiceId']>1].copy()
sg_orders_supplier_count_repeat = sg_orders_supplier_count_repeat.sort_values(by = 'invoiceId', ascending = False)

## Look at top 10/20 most ordered item?
## Remove those with 0 dollar value
top_20_repeat_orders_supplier = sg_orders_supplier_count_repeat.iloc[0:20]
top_20_repeat_orders_supplier.rename(columns = {'invoiceId': 'invoice_count'}, inplace = True)
top20_orders = sg_orders.merge(top_20_repeat_orders_supplier)
top20_orders = top20_orders.loc[top20_orders['pre_tax_amount']>0].copy()

## Min and Max Prices for top 20 orders
top20_orders_min = top20_orders[['productname', 'pre_tax_amount', 'supplierName', 'businessName']].groupby(['productname', 'supplierName', 'businessName']).min().reset_index()
top20_orders_min.rename(columns = {'pre_tax_amount': 'min_pre_tax_amt'}, inplace = True)
top20_orders_max = top20_orders[['productname', 'pre_tax_amount', 'supplierName', 'businessName']].groupby(['productname', 'supplierName', 'businessName']).max().reset_index()
top20_orders_max.rename(columns = {'pre_tax_amount': 'max_pre_tax_amt'}, inplace = True)

## Clear increase in prices
top20_orders_min_max = top20_orders_min.merge(top20_orders_max)
top20_orders_min_max['pct_diff'] = round(100*((top20_orders_min_max['max_pre_tax_amt'] - top20_orders_min_max['min_pre_tax_amt'])/top20_orders_min_max['min_pre_tax_amt']),2)
top20_mean_diff = top20_orders_min_max.groupby(['productname', 'supplierName']).mean(numeric_only = True).reset_index()

# %%
## Determine whether back of house costs have risen month-on-month
monthly_invoice_cost = sg_orders[['invoiceYear', 'invoiceMonth', 'total_pre_tax_amount', 'organizationId']].groupby(['invoiceYear', 'invoiceMonth', 'organizationId']).sum().reset_index()

### Create an index based on Jan 2021 prices
jan21_index = monthly_invoice_cost.loc[((monthly_invoice_cost['invoiceYear'] == 2021) & (monthly_invoice_cost['invoiceMonth'] == 1)), ['organizationId', 'total_pre_tax_amount']].copy().reset_index(drop = True)
jan21_index.rename(columns = {'total_pre_tax_amount': 'jan21_total_pre_tax_amount'}, inplace = True)

monthly_invoice_cost = monthly_invoice_cost.merge(jan21_index)
monthly_invoice_cost['index_value'] = round(monthly_invoice_cost['total_pre_tax_amount']/monthly_invoice_cost['jan21_total_pre_tax_amount'],2)
monthly_invoice_cost['start_date'] = monthly_invoice_cost['invoiceYear'].astype(int).astype(str) + '-' + monthly_invoice_cost['invoiceMonth'].astype(int).astype(str) + '-' + '1'
monthly_invoice_cost['start_date'] = pd.to_datetime(monthly_invoice_cost['start_date'])

# Count the number of months for which we have records
record_count = monthly_invoice_cost[['organizationId', 'start_date']].groupby('organizationId').count().reset_index()
record_count.rename(columns = {'start_date': 'num_of_months'}, inplace = True)
record_count = record_count.loc[record_count['num_of_months'] == 25]

### Index those that have a full record
### It is not obvious if there is an obvious sustained increase in back-of-house costs
complete_monthly_invoice_cost = monthly_invoice_cost.loc[monthly_invoice_cost['organizationId'].isin(record_count.organizationId)]
complete_monthly_invoice_cost = complete_monthly_invoice_cost.loc[complete_monthly_invoice_cost['index_value'] < 3]
sns.lineplot(data = complete_monthly_invoice_cost, x = 'start_date', y = 'index_value', hue = 'organizationId', legend = False)



## Determine cost per invoice
cost_per_invoice = sg_orders[['total_pre_tax_amount', 'invoiceId', 'organizationId', 'supplierName', 'invoiceDate', 'invoiceYear', 'invoiceMonth']].groupby(['invoiceId', 'organizationId', 'supplierName', 'invoiceDate']).sum().reset_index()
cost_per_invoice = cost_per_invoice.loc[cost_per_invoice['total_pre_tax_amount'] > 0].copy()

## Monthly average per supplier per organization
monthly_invoice_cost = cost_per_invoice.groupby(['organizationId', 'supplierName', 'invoiceYear', 'invoiceMonth']).mean(numeric_only = True)
# %%
## Load the Roberta model to simplify the text
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
config = AutoConfig.from_pretrained("C:\\Users\\wesch\\Documents\\FoodRazor\\prises\\tiny_roberta\\models--deepset--tinyroberta-squad2\\snapshots\\20891f44e15bf4a3caf0429fd9319f2632839125\\config.json")
model_name = "deepset/tinyroberta-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

## Define the question
question = 'What is?'

## Set up the pipeline
pipeline = pipeline('question-answering', model = model, tokenizer = tokenizer)

## Function to ask the question
def QnA(question = question, product_name_list = product_name_list):
    answers = []
    for p in product_name_list:
        QnA_input = {'question': question, 'context': p}
        res = pipeline(QnA_input)
        res_df = pd.DataFrame.from_dict(data = res, orient = 'index').reset_index()
        answer = res_df.iloc[3,1]
        answers.append(answer)
    return answers

answers = QnA()

## Append answers to the df
sg_orders_sample['simplified_productname'] = answers
sg_orders_sample = sg_orders_sample[['productname', 'simplified_productname', 'pre_tax_amount', 'quantity', 'invoiceId', 'supplierName', 'businessName', 'organizationId', 'invoiceDate']]

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

