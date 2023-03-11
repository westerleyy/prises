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
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy


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
## Classifying for similarity
unique_cl_product_names = top_orders_df.cl_productname.drop_duplicates()
unique_cl_product_names_df = pd.DataFrame(unique_cl_product_names)

# using tf-idf to vectorize the product name column
vectorizer = TfidfVectorizer(strip_accents = "ascii")
x = vectorizer.fit_transform(unique_cl_product_names)

# %%
## Visualizing the kmeans clusters
sum_squared_distance = []
n_clusters = range(5,30)
for i in n_clusters:
    product_name_kmeans = KMeans(n_clusters = i, random_state = 20).fit(x)
    labels = product_name_kmeans.labels_
    score = metrics.silhouette_score(x, labels, metric = 'euclidean')
    sum_squared_distance.append(product_name_kmeans.inertia_)
    print("n_cluster = " + str(i) + " & silhouette score = " + str(score))
    
plt.plot(n_clusters, sum_squared_distance, 'bx-')
plt.xlabel('No. of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Plot for Product Name Clustering')
plt.show()


# %%
## Assigning clusters of sizes 16, 20, 25
## Entirely arbitrary

product_name_kmeans_k25= KMeans(n_clusters = 25, random_state = 20).fit(x)
product_name_kmeans_k35= KMeans(n_clusters = 35, random_state = 20).fit(x)
product_name_kmeans_k40= KMeans(n_clusters = 40, random_state = 20).fit(x)

unique_cl_product_names_df.loc[:, 'k25']  = product_name_kmeans_k25.labels_
unique_cl_product_names_df.loc[:, 'k35']  = product_name_kmeans_k35.labels_
unique_cl_product_names_df.loc[:, 'k40']  = product_name_kmeans_k40.labels_

unique_cl_product_names_df = unique_cl_product_names_df.merge(top_orders_df[['cl_productname', 'organizationId']]) 

# %%
## Find the max and mean price of each item in the top orders df, the frequency it appears, and the min and max date
## Remove those that cost 0
sg_orders_clean_nonzero = sg_orders_clean.loc[sg_orders_clean['pre_tax_amount']> 0].copy()
sg_orders_price = sg_orders_clean_nonzero[['organizationId', 'pre_tax_amount', 'cl_productname']].groupby(['organizationId', 'cl_productname']).agg(
    min_price = ('pre_tax_amount', 'min'),
    max_price = ('pre_tax_amount', 'max'),
    frequency = ('pre_tax_amount', 'count')
    ).reset_index()
sg_orders_price.drop_duplicates(inplace = True)

grouped_products = unique_cl_product_names_df.merge(sg_orders_price)
grouped_products = grouped_products[['cl_productname', 'organizationId', 'min_price', 'max_price', 'frequency', 'k25', 'k35', 'k40']]
grouped_products.rename(columns = {'cl_productname': 'product_ordered',
                                   'k25': '25_groups',
                                   'k35': '35_groups',
                                   'k40': '40_groups'}, inplace = True)

grouped_products.to_csv('./exports/grouping_with_min_max.csv', index = False)

# %%
## FUCKING DUPLICATES ALL OVER THE PLACE ADJUST WHEN ALF REQUIRES
sg_orders_minprice_date = sg_orders_price[['min_price', 'organizationId', 'cl_productname']].merge(sg_orders_clean_nonzero[['pre_tax_amount', 'cl_productname', 'invoiceDate', 'organizationId']], left_on = ['min_price', 'cl_productname', 'organizationId'], right_on = ['pre_tax_amount', 'cl_productname', 'organizationId'])
sg_orders_minprice_date.sort_values(by = ['cl_productname', 'organizationId', 'invoiceDate'], inplace = True)
sg_orders_minprice_dupes = sg_orders_minprice_date.duplicated()
sg_orders_minprice_date = sg_orders_minprice_date[sg_orders_minprice_dupes]
sg_orders_maxprice_date.drop_duplicates(inplace = True)
sg_orders_minprice_date = sg_orders_minprice_date[['min_price', 'organizationId', 'cl_productname', 'invoiceDate']].copy()
sg_orders_minprice_date.rename(columns = {'invoiceDate': 'min_price_invoiceDate'}, inplace = True)

sg_orders_maxprice_date = sg_orders_price[['max_price', 'organizationId', 'cl_productname']].merge(sg_orders_clean_nonzero[['pre_tax_amount', 'cl_productname', 'invoiceDate', 'organizationId']], left_on = ['max_price', 'cl_productname', 'organizationId'], right_on = ['pre_tax_amount', 'cl_productname', 'organizationId'])
sg_orders_maxprice_date.sort_values(by = ['cl_productname', 'organizationId', 'invoiceDate'], inplace = True)
sg_orders_maxprice_dupes = sg_orders_maxprice_date.duplicated()
sg_orders_maxprice_date = sg_orders_maxprice_date[sg_orders_maxprice_dupes]
sg_orders_maxprice_date.drop_duplicates(inplace = True)
sg_orders_maxprice_date = sg_orders_maxprice_date[['max_price', 'organizationId', 'cl_productname', 'invoiceDate']].copy()
sg_orders_maxprice_date.rename(columns = {'invoiceDate': 'max_price_invoiceDate'}, inplace = True)


sg_orders_price_complete = sg_orders_price.merge(sg_orders_minprice_date)
sg_orders_price_complete = sg_orders_price_complete.merge(sg_orders_maxprice_date, on = ['organizationId', 'cl_productname', 'max_price'])

# %%
## Suppose we take k = 25 and then test for price changes within the group. Look for in-group correlation/similarity
## Weighted Euclidean and Pearson 50/50 or just a simple box plot for tightness of distribution

grouped_cl_product_names_df = unique_cl_product_names_df[['cl_productname', 'k25']].merge(top_orders_df[['cl_productname', 'organizationId']])
grouped_cl_product_names_df.rename(columns = {'k25': 'grouping'}, inplace = True)

### Find the min and max price for each item from the Orders DF
### Remove those that were zero-priced
top_orders_min_max = grouped_cl_product_names_df.merge(sg_orders_clean[['pre_tax_amount', 'organizationId', 'cl_productname']])
no_change = top_orders_min_max.loc[top_orders_min_max['pre_tax_amount'] == 0].copy()
no_change_items = no_change.cl_productname.unique().tolist()
top_orders_min_max = top_orders_min_max.loc[top_orders_min_max['pre_tax_amount'] > 0].copy()
top_orders_min_max = top_orders_min_max.groupby(['organizationId', 'cl_productname']).agg(
    min_pre_tax_amount = ('pre_tax_amount', 'min'),
    max_pre_tax_amount =  ('pre_tax_amount', 'max')
    ).reset_index()
top_orders_min_max['diff'] = top_orders_min_max['max_pre_tax_amount'] - top_orders_min_max['min_pre_tax_amount']
top_orders_min_max['pct_diff'] = round(100*top_orders_min_max['diff']/top_orders_min_max['min_pre_tax_amount'],2)

### merge back grouping info
top_orders_min_max = top_orders_min_max.merge(grouped_cl_product_names_df)

### aggregated price differences within each group
top_orders_grouped_min_max_abs = top_orders_min_max[['grouping', 'diff']].groupby('grouping').agg(
    min_abs_diff = ('diff', 'min'),
    max_abs_diff = ('diff', 'max'),
    mean_abs_diff = ('diff', np.mean),
    median_abs_diff = ('diff', np.median)
    )

top_orders_grouped_min_max_pct = top_orders_min_max[['grouping', 'pct_diff']].groupby('grouping').agg(
    min_pct_diff = ('pct_diff', 'min'),
    max_pct_diff = ('pct_diff', 'max'),
    mean_pct_diff = ('pct_diff', np.mean),
    median_pct_diff = ('pct_diff', np.median)
    )

### pare back outliers
top_orders_min_max_trimmed = top_orders_min_max.loc[top_orders_min_max['pct_diff'] < 300].copy()
top_orders_min_max_abs_trimmed = top_orders_min_max.loc[top_orders_min_max['diff'] < 20].copy()

sns.boxplot(data = top_orders_min_max_trimmed, x = 'grouping', y = 'pct_diff')
sns.boxplot(data = top_orders_min_max_abs_trimmed, x = 'grouping', y = 'diff')

# %%
## Testing for statistical significance of differences in distribution
### Can we run a Kolmogorov Test to determine that the samples for each group are different?

### count the number of items in each group
group_count = top_orders_min_max.groupby('grouping').count()

## 14 and 17 because they are the two biggest groups. 22 as the third
k14 = top_orders_min_max.loc[top_orders_min_max['grouping'] == 14, ['pct_diff']].to_numpy()
k17 = top_orders_min_max.loc[top_orders_min_max['grouping'] == 17, ['pct_diff']].to_numpy()
k22 = top_orders_min_max.loc[top_orders_min_max['grouping'] == 22, ['pct_diff']].to_numpy()
all_orders_pct_diff = top_orders_min_max[['pct_diff']].to_numpy()

### If the KS statistic is large, then the p-value may be small, which may be taken as evid against the null hypothesis
### where the null hypothesis is that the two are the same
### I can prove that some are different from others. But so what? The in-group differences might be pretty big

ks2_obj = scipy.stats.ks_2samp(data1 = np.ravel(k14), data2 = np.ravel(all_orders_pct_diff))
print(ks2_obj.statistic)
print(ks2_obj.pvalue)

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

