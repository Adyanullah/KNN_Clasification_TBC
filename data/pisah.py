import pandas as pd

# Load the dataset
file_path = 'cleaned_data.csv'
data = pd.read_csv(file_path)

# Split the data into training and test sets (80:20 ratio)
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Save the split datasets to CSV files
train_file_path = 'train_data.csv'
test_file_path = 'test_data.csv'
train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

(train_file_path, test_file_path)
