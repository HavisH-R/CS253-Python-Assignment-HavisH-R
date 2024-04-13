import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Binarizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

# Function to preprocess assets and liabilities
def preprocess_assets_liabilities(value):
    if 'Crore' in value:
        return float(value.split()[0])
    elif 'Lac' in value:
        return float(value.split()[0]) / 100
    elif 'Thou' in value:
        return float(value.split()[0]) / 10000
    elif 'Hund' in value:
        return float(value.split()[0]) / 100000
    else:
        return 0

# Binning function for criminal cases
def bin_criminal_cases(number_of_cases):
    if number_of_cases <= 5:
        return 1
    elif number_of_cases <= 15:
        return 2
    else:
        return 3

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data_all = pd.read_csv('test.csv')

# Keep only the required features
required_features = ['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state', 'Education']
train_data = train_data[required_features]
test_data = test_data_all[['Party', 'Criminal Case', 'Total Assets', 'Liabilities', 'state']]

# Label encode 'Education' column
label_encoder = LabelEncoder()
train_data['Education'] = label_encoder.fit_transform(train_data['Education'])

# Preprocess 'Total Assets' and 'Liabilities'
train_data['Total Assets'] = train_data['Total Assets'].apply(preprocess_assets_liabilities)
train_data['Liabilities'] = train_data['Liabilities'].apply(preprocess_assets_liabilities)
test_data['Total Assets'] = test_data['Total Assets'].apply(preprocess_assets_liabilities)
test_data['Liabilities'] = test_data['Liabilities'].apply(preprocess_assets_liabilities)

# Binarizing 'Total Assets' and 'Liabilities'
binarizer = Binarizer()
train_data[['Total Assets', 'Liabilities']] = binarizer.fit_transform(train_data[['Total Assets', 'Liabilities']])
test_data[['Total Assets', 'Liabilities']] = binarizer.transform(test_data[['Total Assets', 'Liabilities']])

# One-hot encoding for 'state' and 'Party'
train_data = pd.get_dummies(train_data, columns=['state', 'Party'])
test_data = pd.get_dummies(test_data, columns=['state', 'Party'])

# Binning 'Criminal Case'
train_data['Criminal Case'] = train_data['Criminal Case'].apply(bin_criminal_cases)
test_data['Criminal Case'] = test_data['Criminal Case'].apply(bin_criminal_cases)

# Binarizing 'Criminal Case' after binning
binarizer = Binarizer(threshold=1.5)  # Threshold to differentiate between low and high criminal cases
train_data['Criminal Case'] = binarizer.fit_transform(train_data[['Criminal Case']])
test_data['Criminal Case'] = binarizer.transform(test_data[['Criminal Case']])

# Separating features and target variable
X = train_data.drop(columns=['Education'])
y = train_data['Education']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Bernoulli Naive Bayes classifier
classifier = BernoulliNB()

# Define the parameter grid to search
param_grid = {
    'alpha': [0.5,0.55,0.6, 0.63, 0.7, 0.72, 0.74, 0.75, 0.76, 0.78, 0.8, 0.85, 0.9, 0.95, 1.0]  
}

# Define F1 score as the scoring metric
scorer = make_scorer(f1_score, average='weighted')

# Initialize GridSearchCV with F1 score as the scoring metric
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring=scorer)

# Perform grid search on training data
grid_search.fit(X_train, y_train)

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on validation data
val_f1_score = grid_search.best_score_
print("Validation F1 Score:", val_f1_score)

# Making predictions on test data
test_predictions = best_model.predict(test_data)
# Converting predictions back to original labels
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Preparing the dataframe for submission
submission_df = pd.DataFrame({'ID': test_data_all['ID'], 'Education': test_predictions_labels})
submission_df.to_csv('predicted_education5.csv', index=False)
