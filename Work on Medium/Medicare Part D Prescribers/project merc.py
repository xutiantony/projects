# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:26:39 2023

@author: xutian
"""



import pandas as pd
df = pd.read_csv("C:/Users/xutian/Downloads/MUP_DPR_RY23_P04_V10_DY21_NPI.CSV")

list(df.columns)
# 0 Overview
total_entries = len(df)
unique_specialties = df['Prscrbr_Type'].nunique()
unique_providers = df['PRSCRBR_NPI'].nunique()


total_claims = df['Tot_Clms'].sum()
avg_claims_per_provider = df.groupby('PRSCRBR_NPI')['Tot_Clms'].sum().mean()
avg_claims_per_specialty = df.groupby('Prscrbr_Type')['Tot_Clms'].sum().mean()
claims_per_provider = df.groupby('PRSCRBR_NPI')['Tot_Clms'].sum()
std_dev_claims_per_provider = claims_per_provider.std()
# Get top 5 specialties with the highest number of claims:
top_specialties = df.groupby('Prscrbr_Type')['Tot_Clms'].sum().sort_values(ascending=False).head(5)

# Printing the results:
print(f"Basic Statistics:\n")
print(f"Total number of rows/entries in the dataset: {total_entries}")
print(f"Number of unique specialties: {unique_specialties}")
print(f"Number of unique providers: {unique_providers}\n")
print(f"Claims:\n")
print(f"Total number of claims: {total_claims}")
print(f"Average number of claims per provider: {avg_claims_per_provider:.2f}")
print(f"Standard deviation of claims per provider: {std_dev_claims_per_provider:.2f}")
print(f"Average number of claims per specialty: {avg_claims_per_specialty:.2f}\n")
print(f"Top 5 specialties with the highest number of claims:\n")
print(top_specialties)

# Costs:
total_cost = df['Tot_Drug_Cst'].sum()
total_generic_drug_cost = df['Gnrc_Tot_Drug_Cst'].sum()
total_brand_drug_cost = df['Brnd_Tot_Clms'].sum()
total_brand_drug_cost_over65 = df['GE65_Tot_Drug_Cst'].sum()
total_brand_drug_cost_low = df['LIS_Drug_Cst'].sum()
avg_cost_per_claim = total_cost / df['Tot_Clms'].sum()
costs_per_provider = df.groupby('PRSCRBR_NPI')['Tot_Drug_Cst'].sum()
avg_cost_per_provider = costs_per_provider.mean()
std_dev_cost_per_provider = costs_per_provider.std()

# Get top 5 specialties with the highest total costs:
top_specialties_by_cost = df.groupby('Prscrbr_Type')['Tot_Drug_Cst'].sum().sort_values(ascending=False).head(5)

# Providers:
unique_providers = df['PRSCRBR_NPI'].nunique()
top_states_by_providers = df['Prscrbr_State_Abrvtn'].value_counts().head(5)
claims_per_provider = df.groupby('PRSCRBR_NPI')['Tot_Clms'].sum()
avg_claims_per_provider = claims_per_provider.mean()
std_dev_claims_per_provider = claims_per_provider.std()

# Printing the results:
print(f"Costs:\n")
print(f"Total cost (for all drugs): ${total_cost:,.2f}")
print(f"Total cost of generic drugs: ${total_generic_drug_cost:,.2f}")
print(f"Total cost of brand name drugs: ${total_brand_drug_cost:,.2f}")
print(f"Total cost of drugs for beneficiaries over 65: ${total_brand_drug_cost_over65:,.2f}")
print(f"Total cost of drugs for beneficiaries with low income subsidy: ${total_brand_drug_cost_low:,.2f}")
print(f"Average cost per claim: ${avg_cost_per_claim:,.2f}")
print(f"Average cost per provider: ${avg_cost_per_provider:,.2f}")
print(f"Standard deviation of cost per provider: ${std_dev_cost_per_provider:,.2f}\n")
print(f"Top 5 specialties with the highest total costs:\n")
print(top_specialties_by_cost)

print(f"\nProviders:\n")
print(f"Number of unique providers: {unique_providers}")
print(f"Top 5 states with the highest number of providers:\n")
print(top_states_by_providers)
print(f"Average number of claims per provider: {avg_claims_per_provider:.2f}")
print(f"Standard deviation of claims per provider: {std_dev_claims_per_provider:.2f}")
print(f"Average cost per provider: ${avg_cost_per_provider:,.2f}")


# 1 EDA
### Provider Entity Segmentation

segmented = df['Prscrbr_Ent_Cd'].value_counts().reset_index()
segmented.columns = ['Prscrbr_Ent_Cd', 'Count']

print(segmented)

import matplotlib.pyplot as plt
import seaborn as sns

merck_green = "#009C63"
sns.set_theme(style="whitegrid", font="Arial", font_scale=1.2)

plt.figure(figsize=(8, 5))
ax = sns.barplot(data=segmented, x='Prscrbr_Ent_Cd', y='Count', palette=[merck_green, '#AAAAAA'])  

ax.set_title('Provider Segmentation by Entity Type', fontsize=16)
ax.set_ylabel('Number of Providers', fontsize=14)
ax.set_xlabel('Entity Type', fontsize=14)
ax.set_xticklabels(['Individual', 'Organization']) 

sns.despine()  # Removes the top and right spines for cleaner look
plt.tight_layout()

plt.show()

### Speciality type
segmented = df['Prscrbr_Type'].value_counts().reset_index()
segmented.columns = ['Prscrbr_Type', 'Count']


mask = segmented['Count'] < 2000
grouped_other = segmented[mask].sum(numeric_only=True)
segmented.loc[mask, 'Prscrbr_Type'] = 'Others'
segmented = segmented.groupby('Prscrbr_Type').sum(numeric_only=True).reset_index()

if 'Others' in grouped_other.index:
    segmented.loc[segmented['Prscrbr_Type'] == 'Others', 'Count'] += grouped_other['Count']

segmented = segmented.sort_values(by='Count', ascending=False)


others_row = segmented[segmented['Prscrbr_Type'] == 'Others']
segmented = segmented[segmented['Prscrbr_Type'] != 'Others']
segmented = pd.concat([segmented, others_row])

plt.figure(figsize=(12, 15))
ax = sns.barplot(data=segmented, y='Prscrbr_Type', x='Count',palette=[merck_green, '#AAAAAA'])

# Adding count labels to each bar
for index, value in enumerate(segmented['Count']):
    ax.text(value, index, str(value), color='black', ha="left", va="center", fontsize=10)

plt.title('Provider Type Segmentation', fontsize=18)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Provider Type', fontsize=14)
plt.show()


### Provider Gender Segmentation

segmented = df['Prscrbr_Gndr'].value_counts().reset_index()
segmented.columns = ['Prscrbr_Gndr', 'Count']

print(segmented)

gender_counts = df.groupby('Prscrbr_Type')['Prscrbr_Gndr'].value_counts().unstack().fillna(0)

gender_distribution = df.groupby('Prscrbr_Type')['Prscrbr_Gndr'].value_counts(normalize=True).mul(100).unstack().fillna(0)

combined_df = gender_counts.join(gender_distribution, lsuffix='_Count', rsuffix='_Percentage')

combined_df.to_csv('gender_distribution_by_specialty.csv')

print(combined_df)

## Geographical distribution
cn_counts = df['Prscrbr_Cntry'].value_counts().reset_index()
cn_counts.columns = ['Country', 'Number_of_Providers']

import plotly.express as px


state_counts = df.loc[df['Prscrbr_Cntry'] == 'US']['Prscrbr_State_Abrvtn'].value_counts().reset_index()
state_counts.columns = ['State', 'Number_of_Providers']

fig = px.choropleth(state_counts, 
                    locations='State', 
                    color='Number_of_Providers',
                    hover_name='State',
                    hover_data=['Number_of_Providers'],
                    locationmode='USA-states',
                    color_continuous_scale='viridis',
                    scope="usa",
                    title="Number of Providers by State")

fig.write_html("path_to_save_figure.html")


    ### Claims by state
us_df = df[df['Prscrbr_Cntry'] == 'US']

state_claims = us_df.groupby('Prscrbr_State_Abrvtn')['Tot_Clms'].sum().reset_index()
state_claims.columns = ['State', 'Total_Claims']

fig = px.choropleth(state_claims, 
                    locations='State', 
                    color='Total_Claims',
                    hover_name='State',
                    hover_data=['Total_Claims'],
                    locationmode='USA-states',
                    color_continuous_scale='viridis',
                    scope="usa",
                    title="Total Number of Claims by State")

fig.write_html("claims.html")

    ### Total Cost by state
us_df = df[df['Prscrbr_Cntry'] == 'US']

state_claims = us_df.groupby('Prscrbr_State_Abrvtn')['Tot_Drug_Cst'].sum().reset_index()
state_claims.columns = ['State', 'Tot_Drug_Cst']

fig = px.choropleth(state_claims, 
                    locations='State', 
                    color='Tot_Drug_Cst',
                    hover_name='State',
                    hover_data=['Tot_Drug_Cst'],
                    locationmode='USA-states',
                    color_continuous_scale='viridis',
                    scope="usa",
                    title="Total Drug Cost by State")

fig.write_html("costs.html")

## Total claims
df['Tot_Clms'].describe()

plt.figure(figsize=(10, 6))

import numpy as np

# Apply log transformation
df['log_Tot_Clms'] = np.log(df['Tot_Clms'])

plt.figure(figsize=(10, 6))
sns.histplot(df['log_Tot_Clms'], kde=False, bins=50,color='#009966')
plt.title('Log-transformed Distribution of Total Claims')
plt.xlabel('Log of Total Claims')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

ddd= df.loc[df['Tot_Clms'] >=240000]
#2. Drug Segmentation
## Opioid Percentage

total_claims_by_state = df.groupby('Prscrbr_State_Abrvtn')['Tot_Clms'].sum()
opioid_claims_by_state = df.groupby('Prscrbr_State_Abrvtn')['Opioid_Tot_Clms'].sum()

# Calculate the opioid claims percentage
opioid_percentage_by_state = (opioid_claims_by_state / total_claims_by_state) * 100
opioid_percentage_df = opioid_percentage_by_state.reset_index()
opioid_percentage_df.columns = ['State', 'Opioid_Percentage']

import plotly.express as px

fig = px.choropleth(opioid_percentage_df, 
                    locations='State', 
                    color='Opioid_Percentage',
                    hover_name='State',
                    hover_data=['Opioid_Percentage'],
                    locationmode='USA-states',
                    color_continuous_scale='viridis',
                    scope="usa",
                    title="Opioid Claims Percentage by State")

fig.write_html("Opioid.html")

# brand name drug

total_claims_by_state = df.groupby('Prscrbr_State_Abrvtn')['Tot_Drug_Cst'].sum()
opioid_claims_by_state = df.groupby('Prscrbr_State_Abrvtn')['Brnd_Tot_Drug_Cst'].sum()

# Calculate the opioid claims percentage
opioid_percentage_by_state = (opioid_claims_by_state / total_claims_by_state) * 100
opioid_percentage_df = opioid_percentage_by_state.reset_index()
opioid_percentage_df.columns = ['State', 'Brand Name drug cost percentage']

import plotly.express as px

fig = px.choropleth(opioid_percentage_df, 
                    locations='State', 
                    color='Brand Name drug cost percentage',
                    hover_name='State',
                    hover_data=['Brand Name drug cost percentage'],
                    locationmode='USA-states',
                    color_continuous_scale='viridis',
                    scope="usa",
                    title="Brand Name drug cost percentage by State")

fig.write_html("Brand Name drug cost percentage.html")

grouped_df = df.groupby('Prscrbr_Type')['Tot_Drug_Cst', 'Brnd_Tot_Drug_Cst'].sum().reset_index()

grouped_df.to_csv('specialty_drug_costs.csv', index=False)


# cost segmentation

grouped = df.groupby('Prscrbr_Type').agg({
    'Tot_Clms': 'sum',
    'Opioid_Tot_Clms': 'sum',
    'Antbtc_Tot_Clms': 'sum',
    'Antpsyct_GE65_Tot_Clms': 'sum'
}).reset_index()

grouped['Total'] = grouped['Tot_Clms']

grouped['Others'] = (grouped['Total'] - grouped['Opioid_Tot_Clms']
                     - grouped['Antbtc_Tot_Clms']
                     - grouped['Antpsyct_GE65_Tot_Clms'])

grouped = grouped.sort_values(by='Total', ascending=False)

top_30 = grouped.iloc[:30].copy()
rest = grouped.iloc[30:].sum(numeric_only=True)
rest['Prscrbr_Type'] = 'Others'
top_30 = top_30.append(rest, ignore_index=True)

fig, ax = plt.subplots(figsize=(16, 15))
fig, ax = plt.subplots(figsize=(16, 15))
top_30.set_index('Prscrbr_Type')[['Opioid_Tot_Clms', 'Antbtc_Tot_Clms', 'Antpsyct_GE65_Tot_Clms', 'Others']].plot(kind='barh', stacked=True, color=sns.color_palette("viridis", 5), ax=ax)
ax.set_xlabel('Total Claims')
ax.set_title('Total Claims by Specialty for Top 30 Specialties')
plt.tight_layout()
plt.show()

# 3. Beneficiary 
######by age
age_groups = {
    'Bene_Age_LT_65_Cnt': 'Less Than 65',
    'Bene_Age_65_74_Cnt': '65 to 74',
    'Bene_Age_75_84_Cnt': '75 to 84',
    'Bene_Age_GT_84_Cnt': 'Greater Than 84'
}

age_sums = {value: df[key].sum() for key, value in age_groups.items()}

age_df = pd.DataFrame(list(age_sums.items()), columns=['Age Group', 'Count'])

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=age_df, y='Age Group', x='Count', palette="viridis")

ax.set_xticklabels(['{:.0f}M'.format(x/1e6) for x in ax.get_xticks()])
for index, value in enumerate(age_df['Count']):
    ax.text(value, index, '  {:.1f}M'.format(value/1e6))
    
plt.title('Beneficiary Count by Age Group', fontsize=18)
plt.xlabel('Count (in Millions)', fontsize=14)
plt.ylabel('Age Group', fontsize=14)
plt.tight_layout()
plt.show()

######by gender
age_groups = {
    'Bene_Feml_Cnt': 'Female group',
    'Bene_Male_Cnt': 'Male group'
}

age_sums = {value: df[key].sum() for key, value in age_groups.items()}

age_df = pd.DataFrame(list(age_sums.items()), columns=['Gender Group', 'Count'])

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=age_df, y='Gender Group', x='Count', palette="viridis")

ax.set_xticklabels(['{:.0f}M'.format(x/1e6) for x in ax.get_xticks()])

for index, value in enumerate(age_df['Count']):
    ax.text(value, index, '  {:.1f}M'.format(value/1e6))
    
plt.title('Beneficiary Count by Gender Group', fontsize=18)
plt.xlabel('Count (in Millions)', fontsize=14)
plt.ylabel('Age Group', fontsize=14)
plt.tight_layout()
plt.show()
##### by race

race_groups = {
    'Bene_Race_Wht_Cnt': 'Non-Hispanic White',
    'Bene_Race_Black_Cnt': 'Black/African American',
    'Bene_Race_Api_Cnt': 'Asian Pacific Islander',
    'Bene_Race_Hspnc_Cnt': 'Hispanic',
    'Bene_Race_Natind_Cnt': 'American Indian/Alaskan Native',
    'Bene_Race_Othr_Cnt': 'Race Not Elsewhere Classified'
}

race_sums = {value: df[key].sum() for key, value in race_groups.items()}

race_df = pd.DataFrame(list(race_sums.items()), columns=['Race', 'Count'])

total_beneficiaries = race_df['Count'].sum()
race_df['Percentage'] = (race_df['Count'] / total_beneficiaries) * 100

plt.figure(figsize=(12, 7))
ax = sns.barplot(data=race_df, y='Race', x='Count', palette="viridis")

ax.set_xticklabels(['{:.0f}M'.format(x/1e6) for x in ax.get_xticks()])

for index, (value, percentage) in enumerate(zip(race_df['Count'], race_df['Percentage'])):
    ax.text(value, index, '  {:.1f}M ({:.1f}%)'.format(value/1e6, percentage))

plt.title('Beneficiary Count by Race', fontsize=18)
plt.xlabel('Count (in Millions)', fontsize=14)
plt.ylabel('Race', fontsize=14)
plt.tight_layout()
plt.show()

#############################Regression analysis################
import pandas as pd
data = pd.read_csv("C:/Users/xutian/Desktop/project merc/path_to_your_output_file.CSV")


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dummifying categorical variables
categorical_columns = ['Prscrbr_Gndr', 'Prscrbr_State_Abrvtn', 'Prscrbr_RUCA_Desc', 'Prscrbr_Type']
encoder = OneHotEncoder(sparse=False, drop='first') # drop first to avoid multicollinearity
encoded_data = encoder.fit_transform(data[categorical_columns])

# Creating a new DataFrame with encoded categorical data and existing numerical data
encoded_columns = encoder.get_feature_names(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Combining the encoded columns with the rest of the data
numerical_data = data.drop(columns=categorical_columns)
processed_data = pd.concat([numerical_data, encoded_df], axis=1)

# Removing outliers
# Define a function to remove outliers based on the Z-score
def remove_outliers(df, threshold=5):
    z_scores = np.abs(df.select_dtypes(include=[np.number]).apply(lambda x: (x - x.mean()) / x.std()))
    return df[(z_scores < threshold).all(axis=1)]

cleaned_data = remove_outliers(processed_data)


# Splitting the data into features and target
X = cleaned_data.drop('Tot_Clms', axis=1)
y = cleaned_data['Tot_Clms']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and performance evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
coefficients = model.coef_
intercept = model.intercept_


# Visualizing results
plt.figure(figsize=(10, 6))
sns.regplot(y_test, y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('Actual Total Claims')
plt.ylabel('Predicted Total Claims')
plt.title('Actual vs Predicted Total Claims')

# Return the regression results
regression_results = {
    'Mean Squared Error': mse,
    'R-squared': r2,
    'Coefficients': coefficients,
    'Intercept': intercept
}

regression_results, plt.show()

# Extracting coefficients with positive values
positive_coefficients = {encoded_columns[i]: coefficients[i] for i in range(len(coefficients)) if coefficients[i] != 0}

positive_coefficients

























