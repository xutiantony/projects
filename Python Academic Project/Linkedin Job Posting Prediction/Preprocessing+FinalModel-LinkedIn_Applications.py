# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:58:46 2023

@author: Tony
"""

# Importing Packages
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import wordnet
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
import shap
from sklearn.metrics import roc_curve, auc


# Load the data
df = pd.read_csv("C:/Users/Tony/Downloads/linkdin_Job_data.csv")
df = df.drop(columns=[ 'company_id','hiring_person_link','alumni'])
###Task 1: Data Preprocessing
# 1.Drop job id and details id
df=df.drop_duplicates(subset='job_ID', keep='first')

# 2. keep hours since posted
#dummify hiring name given or not column
df['Hiring_person_indicator'] = df['Hiring_person'].notna().astype(int)

# 3.Convert time since posted to common hours

# Function to convert values to hours
def to_hours(value):
    # Check if the value is a string type
    if isinstance(value, str):
        if "minute" in value:
            return int(value.split()[0]) / 60
        elif "day" in value:
            return int(value.split()[0]) * 24
        elif "hour" in value:
            return int(value.split()[0])
    return value

df['posted_day_ago'] = df['posted_day_ago'].apply(to_hours)

print(df)

# 4.Applications/hours for application rate

# Convert no_of_application column to float
df['no_of_application'] = pd.to_numeric(df['no_of_application'], errors='coerce')
df['posted_day_ago'] = pd.to_numeric(df['posted_day_ago'], errors='coerce')

# Calculate Application_rate
df['Application_rate'] = df['no_of_application'] / df['posted_day_ago']

# 5.Create column as 1 or 0 based on above avg or not

# Calculate average of scores
avg_score = df['Application_rate'].mean()

# Create new column 'above_avg'
df['above_avg'] = np.where(df['Application_rate'] > avg_score, 1, 0)


#6.	Also add len(job_details) because maybe longer job descriptions get more applications
df['len_job_details'] = df['job_details'].apply(lambda x: len(x) if isinstance(x, str) else 0)

unique_locations_list = df['location'].unique().tolist()
print(unique_locations_list)

# 7.	Extract the state from the location column
states = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
    'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana',
    'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal', 'andaman and nicobar islands',
    'chandigarh', 'dadra and nagar haveli', 'daman and diu', 'delhi', 'lakshadweep',
    'puducherry', 'jammu & kashmir'  # Added 'Jammu & Kashmir'
]

# Function to find the state in the location
def find_state(location):
    if pd.notnull(location):  # Check if 'location' is not NaN
        location_lower = location.lower()
        for state in states:
            if state in location_lower:
                return state.title()
            elif 'bengaluru' in location_lower or 'bangalore' in location_lower:
                 return 'Karnataka'
            elif 'pune' in location_lower or 'mumbai' in location_lower or 'nagpur' in location_lower:
                 return 'Maharashtra'
            elif 'chennai' in location_lower or 'coimbatore' in location_lower:
                 return 'Tamil Nadu'
            elif 'vadodara' in location_lower or 'ahmedabad' in location_lower:
                 return 'Gujarat'
            elif 'kolkata' in location_lower:
                 return 'West Bengal'
            elif 'hyderabad' in location_lower:
                 return 'Telangana'
    return 'Multi-state Jobs'


# 8.Apply the function to create the new 'state' column
df['state'] = df['location'].apply(find_state)

print(df[['location', 'state']])


# Assuming df is your DataFrame with 'location' and 'state' columns already present
multi_state_locations = df[df['state'] == 'Multi-state Jobs']['location'].unique()

# Convert NumPy array to a list
multi_state_locations_list = multi_state_locations.tolist()

print(multi_state_locations_list)


#  9. download the WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample dataframe
# df = pd.read_csv("C:/Users/vinay/OneDrive/Documents/.spyder-py3/job_cleanData.csv")

# Define keywords and their synonyms for each category
categories = {
    'Benefits-Related': ["bonuses","awards", "healthcare", "retirement", "vacation","flexible","outings"],
    'Company-Culture': ["inclusive", "diverse","balance", "collaborative","innovative",'diversity','equity','inclusion',"teamwork",'people'],
    'Growth and Development':["career","grow","training","development","mentorship","learning","self-development"]
}

# Fetch synonyms for the keywords
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Replace underscore with space for multi-word synonyms
    return list(synonyms)

# Create a dictionary with keywords and their synonyms
category_synonyms = {}
for category, keywords in categories.items():
    synonyms = set()
    for keyword in keywords:
        synonyms.update([keyword])        # Add the keyword itself
        synonyms.update(get_synonyms(keyword))  # Add the synonyms of the keyword
    category_synonyms[category] = synonyms

# Create dummy variables
df['job_details'] = df['job_details'].astype(str)
for category, synonyms in category_synonyms.items():
    df[category] = df['job_details'].apply(lambda x: 1 if any(syn in x for syn in synonyms) else 0)


# 10. Create a column for the number of years of experience required

def experience_to_number(text):
    patterns = [
        r'(\d+)(?:-|\s?to\s?)(\d+)\syears',
        r'(\d+)\s?\+\s?years',
        r'(?:at least\s)?(\d+)\syears',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return max(int(num) for num in match.groups() if num is not None)
    return None

df['experience_num'] = df['job_details'].apply(lambda x: experience_to_number(str(x)))

degree_mapping = {
    'bachelor': ['bachelor', 'b.sc', 'b.s.', 'bs', 'undergraduate'],
    'master': ['master', 'm.sc', 'm.s.', 'ms', 'graduate'],
    'phd': ['phd', 'ph.d.', 'doctorate']
}

def create_degree_dummies(text, degree_mapping):
    text = text.lower()
    dummies = {degree: 0 for degree in degree_mapping.keys()}

    for degree, synonyms in degree_mapping.items():
        if any(synonym in text for synonym in synonyms):
            dummies[degree] = 1

    return pd.Series(dummies)

degree_dummies = df['job_details'].apply(lambda x: create_degree_dummies(str(x), degree_mapping))
df = df.join(degree_dummies)


experience_level = ["Mid-Senior level", "Associate", "Entry level", "Executive", "Director", "Internship"]
job_type = ["Full-time", "Contract", "Internship", "Part-time", "Temporary"]
df['Experience Level'] = df["full_time_remote"].apply(lambda x: next((level for level in experience_level if isinstance(x, str) and level in x), np.nan))
df['Job Type'] = df["full_time_remote"].apply(lambda x: next((job for job in job_type if isinstance(x, str) and job in x), np.nan))

# Extract the number of employees from the no_of_employ column
df['employee_range'] = df['no_of_employ'].str.extract(r'(.*?)\s+employees')
# Extract the industry name similarly from the no_of_employ column
df['industry'] = df['no_of_employ'].str.extract(r'·\s+(.*)')

# 11.Map industry to a predefined category

def map_industry_to_category(industry):
    
    if isinstance(industry, str):
    # IT and Software
        if 'it' in industry.lower() or 'software' in industry.lower() or 'technology' in industry.lower() or 'computer' in industry.lower() or 'internet' in industry.lower() or 'data' in industry.lower():
            return 'IT & Software'

        # Engineering and Manufacturing
        elif 'engineering' in industry.lower() or 'manufacturing' in industry.lower() or 'industrial' in industry.lower() or 'machinery' in industry.lower() or 'semiconductor' in industry.lower():
            return 'Engineering & Manufacturing'

        # Healthcare
        elif 'health' in industry.lower() or 'medical' in industry.lower() or 'pharmaceutical' in industry.lower() or 'biotechnology' in industry.lower():
            return 'Healthcare'

        # Finance and Business
        elif 'financial' in industry.lower() or 'business' in industry.lower() or 'accounting' in industry.lower() or 'banking' in industry.lower() or 'investment' in industry.lower():
            return 'Finance & Business'

        # Education and Research
        elif 'education' in industry.lower() or 'research' in industry.lower() or 'e-learning' in industry.lower() or 'training' in industry.lower():
            return 'Education & Research'

        # Media, Marketing, and Communications
        elif 'media' in industry.lower() or 'marketing' in industry.lower() or 'advertising' in industry.lower() or 'publishing' in industry.lower() or 'communications' in industry.lower():
            return 'Media, Marketing & Communications'

        # Retail and Consumer Services
        elif 'retail' in industry.lower() or 'consumer' in industry.lower() or 'apparel' in industry.lower() or 'luxury' in industry.lower() or 'fashion' in industry.lower():
            return 'Retail & Consumer Services'

        # Energy and Environment
        elif 'energy' in industry.lower() or 'environmental' in industry.lower() or 'oil' in industry.lower() or 'gas' in industry.lower() or 'renewable' in industry.lower():
            return 'Energy & Environment'

        # Transportation and Logistics
        elif 'transportation' in industry.lower() or 'logistics' in industry.lower() or 'aviation' in industry.lower() or 'automotive' in industry.lower() or 'maritime' in industry.lower():
            return 'Transportation & Logistics'

        # Telecommunications
        elif 'telecom' in industry.lower() or 'wireless' in industry.lower():
            return 'Telecommunications'

        # Government and Public Services
        elif 'government' in industry.lower() or 'public' in industry.lower() or 'administration' in industry.lower() or 'military' in industry.lower():
            return 'Government & Public Services'

        # Other
        else:
            return 'Other'

# Apply the mapping function to the industry column to create a new 'industry_category' column
df['industry_category'] = df['industry'].apply(map_industry_to_category)

# Rank the number of employees according to business logic. Rank 4 means very large company and 1 means small.

def map_employee_range_to_rank(range):
    if pd.isna(range):
        return None  
    if isinstance(range, float): 
        return None  
    if isinstance(range, str):
        parts = range.replace('+', '').split('-')
        parts = [int(part.replace(',', '')) for part in parts]
        low = parts[0]
        high = parts[-1] 

        if low <= 50:
            return 1  # Small companies
        elif 51 <= low <= 500:
            return 2  # Medium companies
        elif 501 <= low <= 5000:
            return 3  # Large companies
        else:
            return 4  # Very large companies
    else:
        return None  # for any other unexpected types

# Apply the mapping function to the 'employee_range' column
df['rank'] = df['employee_range'].apply(map_employee_range_to_rank)

print(df)

df.to_csv('C:/Users/Tony/Downloads/out3.csv')

# ### Add different data roles
df['data_analyst'] = df['job'].str.contains('data analyst', case=False, na=False) 
df['data_engineer'] = df['job'].str.contains('data engineer', case=False, na=False) 
df['data_scientist'] = df['job'].str.contains('data scientist', case=False, regex=True, na=False) 
df['backend'] = df['job'].str.contains('backend', case=False, na=False)


# ### Columns to drop
columns_to_drop = ['Unnamed: 0','Unnamed: 0.1', 'job_ID', 'job', 'location', 'company_name', 
                   'work_type', 'full_time_remote', 'no_of_employ', 'no_of_application', 
                   'posted_day_ago', 'Hiring_person', 'processed_description', 
                   'job_details', 'Application_rate', 'Column1']

df = df.drop(columns=columns_to_drop)

# ### Additional Preprocessing- for model building
# 
# 1. LinkedIn followers

df['linkedin_followers'] = df['linkedin_followers'].str.replace(',', '').str.replace(' followers', '')
df['linkedin_followers'] = pd.to_numeric(df['linkedin_followers'], errors='coerce').fillna(0).astype(int)
mean_followers = df.loc[df['linkedin_followers'] != 0, 'linkedin_followers'].mean()

# Imputing mean
df.loc[df['linkedin_followers'] == 0, 'linkedin_followers'] = mean_followers

##FINAL SELECTED MODEL CODE - RANDOM FOREST

# 2. Dummifying
categorical = df.select_dtypes(include=['object', 'category']).columns

# Dummify only categorical variables
df_dummies = pd.get_dummies(df[categorical], drop_first=True)
df = df.drop(categorical, axis=1)
df = pd.concat([df, df_dummies], axis=1)

# 3. Splitting to X and y
# Split the data into features and target
X = df.drop('above_avg', axis=1)
y = df['above_avg']

# 4. Imputing missing values`

imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)  



# 5.Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)



# 6.Initialize SMOTE
smote = SMOTE()

 

# Fit and apply SMOTE only on training data
X_train, y_train = smote.fit_resample(X_train, y_train)

# 7.Hyperparameters for GridSearchCV 

param_grid = {
    'n_estimators': [50,100, 150, 200, 250],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        # Maximum depth of the trees
    'min_samples_split': [2, 4, 6, 8],      # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2,3, 4, 6],       # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 0.5, 0.75,3,4,5,6]  # Number of features to consider for the best split
}

# 8.Initialize the GridSearchCV with the Random Forest model
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Using the best estimator found by GridSearchCV
model_rf = grid_search.best_estimator_

# Get probability scores for the precision-recall curve
probabilities = model_rf.predict_proba(X_test)[:, 1]

# Calculate precision-recall pairs for different probability thresholds
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Choose a new threshold for classification
new_threshold = 0.30
modified_predictions = (probabilities >= new_threshold).astype(int)

# Evaluate the model with the modified predictions
new_accuracy = accuracy_score(y_test, modified_predictions)
new_cm = confusion_matrix(y_test, modified_predictions)
new_report = classification_report(y_test, modified_predictions)

# Output the evaluation metrics
print("\nConfusion Matrix:\n", new_cm)
print("\nClassification Report:\n", new_report)
print("\nBest hyperparameters from GridSearchCV:")
print(grid_search.best_params_)


#  Feature Importance Plot
# Calculate feature importances and sort them
feature_importances = model_rf.feature_importances_
sorted_idx = feature_importances.argsort()


X_df = pd.DataFrame(X)

# Take only the top 20 features
top_n = 20
top_sorted_idx = sorted_idx[-top_n:]

plt.figure(figsize=(10, 6))
plt.barh(X_df.columns[top_sorted_idx], feature_importances[top_sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Top 20 Features")
plt.show()


# 3. SHAP Values 
explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(X_train)

# Summarize the effects of all the features
shap.summary_plot(shap_values, X_df, plot_type="bar")



# Assuming pred_probs contains the predicted probabilities for the positive class
pred_probs = model_rf.predict_proba(X_test)[:, 1]

# Compute ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


###APPENIDIX -OTHER MODELS Tested (Snippets) #################################################################################################
#Gradient Boosting Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_imputed, y)  # Assuming y is your target variable
important_features = rf.feature_importances_ > 0

X_selected = X_imputed[:, important_features]

# Create a DataFrame with the features and their importance scores
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
csv_file_path = "C:/Users/vinay/Downloads/imp_features.csv"  # e.g., 'C:/Users/YourName/Documents/importance_data.csv'

# Save the DataFrame to a CSV file
importance_df.to_csv(csv_file_path, index=False)  # Set index=False if you don't want to save the DataFrame index

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3)

# Initialize SMOTE
smote = SMOTE()

 

# Fit and apply SMOTE only on training data
X_train, y_train = smote.fit_resample(X_train, y_train)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2,3, 4, 5,6,7],
    'learning_rate': [0.01,0.05,0.1,0.15,0.2],
    'n_estimators': [100, 200, 300,400,500]
}

grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy',n_jobs=-1)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

from sklearn.metrics import accuracy_score

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#KNN
# standardize
categorical_list = []

for column in X.columns:
    unique_values = X[column].unique()
    if set(unique_values) == {0, 1}:
        categorical_list.append(column)
        
categorical_X = X[categorical_list]

numerical_X = X.drop(columns = categorical_list, axis = 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_X_std = scaler.fit_transform(numerical_X)
numerical_X_std = pd.DataFrame(numerical_X_std, columns =numerical_X.columns) 

X_std = pd.merge(numerical_X_std, categorical_X, left_index=True, right_index=True)

nan_mask = X_std.isna().any(axis=1)

X_std = X_std[~nan_mask]
y = y[~nan_mask]

# test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.33,random_state=5)


# feature selection
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso_model = lasso.fit(X_std, y)

feature_selection = pd.DataFrame(list(zip(X.columns, lasso_model.coef_)), columns=['predictor', 'coefficient'])

# Extracting the coefficients
coef = pd.Series(lasso_model.coef_ != 0, index=X.columns) 

# Selecting non-zero coefficients
selected_features = coef[coef].index

# Use only selected features for model fitting
X_train_knn, X_test_knn, y_train, y_test = train_test_split(X_std[selected_features], y, test_size=0.3, random_state=5)

from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': range(10,26),
    'metric': ['euclidean', 'manhattan', 'chebyshev']
    }

knn = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_knn, y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_model = grid_search.best_estimator_

print(best_params, best_accuracy, best_model)


#Logistic Regression
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)  # Assuming X is your feature dataframe

### Standardize
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
scaled_X = standardizer.fit_transform(X_imputed)

### Train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.33, random_state = 5)


### Feature selection with Lasso
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l1', solver='liblinear')
logreg.fit(scaled_X, y)

# Extracting the coefficients
coef = pd.Series(logreg.coef_[0], index=X.columns)

# Selecting non-zero coefficients
selected_features = coef[coef != 0].index

# Split the imputed and standardized data with only selected features
X_selected = X_imputed[:, X.columns.get_indexer(selected_features)]
X_train_lasso, X_test_lasso, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=0)

# Fit the Logistic Regression model on the selected features
Logistic_Model = logreg.fit(X_train_lasso, y_train)

# Create a DataFrame with the features and their coefficients
coefficients_df = pd.DataFrame(list(zip(selected_features, Logistic_Model.coef_[0])), columns=['predictor', 'coefficient'])

# Sort the DataFrame by the absolute value of the coefficients to get the significance
coefficients_df['abs_coefficient'] = coefficients_df['coefficient'].abs()
sorted_coefficients_df = coefficients_df.sort_values(by='abs_coefficient', ascending=False)

# Drop the absolute value column if you don't want to display it
sorted_coefficients_df.drop(columns=['abs_coefficient'], inplace=True)

# Print the sorted DataFrame
print(sorted_coefficients_df)

### Evaluate model
y_test_pred = Logistic_Model.predict(X_test_lasso)

from sklearn import metrics
metrics.accuracy_score(y_test, y_test_pred)
metrics.precision_score(y_test, y_test_pred)
metrics.recall_score(y_test, y_test_pred)
metrics.f1_score(y_test, y_test_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


#MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()
X_1 = scaler.fit_transform(X_imputed)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE()

# Fit and apply SMOTE only on training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# Scale the features (important for neural networks)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(max_iter=1000)

# Create GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
grid_search.fit(X_train_smote, y_train_smote)

# Best parameters
print(grid_search.best_params_)
# Create MLPClassifier
mlp = MLPClassifier(activation='relu', alpha= 0.0001, hidden_layer_sizes=(100,), 
                    learning_rate='constant', 
                    solver='adam', max_iter=1000)
# Train the model
mlp.fit(X_train_smote, y_train_smote)
# Make predictions
predictions = mlp.predict(X_test)
threshold = 0.30
y_pred_adjusted = (mlp.predict_proba(X_test)[:, 1] >= threshold).astype(int)
# Evaluate the model
print(confusion_matrix(y_test, y_pred_adjusted))
print(classification_report(y_test, y_pred_adjusted))

