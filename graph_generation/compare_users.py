import pandas as pd

base_data_path = "data/"
# Load the CSV files
train_df = pd.read_csv(f"{base_data_path}/interactions_train.csv")
validation_df = pd.read_csv(f"{base_data_path}/interactions_validation.csv")
test_df = pd.read_csv(f"{base_data_path}/interactions_test.csv")

# Get unique user IDs from each dataset
train_user_ids = set(train_df['user_id'].unique())
validation_user_ids = set(validation_df['user_id'].unique())
test_user_ids = set(test_df['user_id'].unique())

# Check if all user IDs in validation and test sets are also in the training set
validation_ids_in_train = validation_user_ids.issubset(train_user_ids)
test_ids_in_train = test_user_ids.issubset(train_user_ids)

# Print the results
print("All validation user IDs in training set:", validation_ids_in_train)
print("All test user IDs in training set:", test_ids_in_train)

# If you need to see which IDs are missing
missing_validation_ids = validation_user_ids - train_user_ids
missing_test_ids = test_user_ids - train_user_ids

print("Missing validation user IDs:", missing_validation_ids)
print("Missing test user IDs:", missing_test_ids)
