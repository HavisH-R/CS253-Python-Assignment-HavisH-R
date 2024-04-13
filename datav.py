import matplotlib.pyplot as plt
import pandas as pd

# Load train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine train and test datasets
combined_data = pd.concat([train_data, test_data])

# Identify candidates with the most criminal records (top 10%)
top_10_percent = combined_data['Criminal Case'].quantile(0.9)
most_criminal_candidates = combined_data[combined_data['Criminal Case'] >= top_10_percent]

# Calculate percentage distribution of parties with candidates having the most criminal records
party_distribution = most_criminal_candidates['Party'].value_counts(normalize=True) * 100

# Plot histogram
plt.figure(figsize=(10, 6))
party_distribution.plot(kind='bar', color='skyblue')
plt.title('Percentage Distribution of Parties with Candidates having the Most Criminal Records (Top 10%)')
plt.xlabel('Political Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
