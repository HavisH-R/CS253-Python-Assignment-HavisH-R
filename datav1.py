import matplotlib.pyplot as plt
import pandas as pd

# Function to preprocess assets and liabilities
def preprocess_assets_liabilities(value):
    if 'Crore' in value:
        return float(value.split()[0]) * 10000000
    elif 'Lac' in value:
        return float(value.split()[0]) * 100000
    elif 'Thou' in value:
        return float(value.split()[0]) * 1000
    else:
        return 0

# Load train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Apply preprocessing to 'Total Assets' column
train_data['Total Assets'] = train_data['Total Assets'].apply(preprocess_assets_liabilities)
test_data['Total Assets'] = test_data['Total Assets'].apply(preprocess_assets_liabilities)

# Combine train and test datasets
combined_data = pd.concat([train_data, test_data])

# Identify candidates with the most wealth (top 10%)
top_10_percent_wealth = combined_data['Total Assets'].quantile(0.9)
wealthiest_candidates = combined_data[combined_data['Total Assets'] >= top_10_percent_wealth]

# Calculate percentage distribution of parties with the most wealthy candidates
wealthy_party_distribution = wealthiest_candidates['Party'].value_counts(normalize=True) * 100

# Plot histogram for wealthiest candidates
plt.figure(figsize=(10, 6))
wealthy_party_distribution.plot(kind='bar', color='lightgreen')
plt.title('Percentage Distribution of Parties with the Most Wealthy Candidates (Top 10%)')
plt.xlabel('Political Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
