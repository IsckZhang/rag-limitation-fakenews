import pandas as pd

# 1. Load data, extracting only the "text" column
df_fake_1 = pd.read_csv("Fake.csv", usecols=["text"])
df_fake_2 = pd.read_csv("DataSet_Misinfo_FAKE.csv", usecols=["text"])
df_true_1 = pd.read_csv("True.csv", usecols=["text"])
df_true_2 = pd.read_csv("DataSet_Misinfo_TRUE.csv", usecols=["text"])

# 2. Label the data (0 = fake, 1 = true)
df_fake_1["label"] = 0
df_fake_2["label"] = 0
df_true_1["label"] = 1
df_true_2["label"] = 1

# 3. Combine and shuffle
df_combined = pd.concat([df_fake_1, df_fake_2, df_true_1, df_true_2], ignore_index=True)
df_shuffled = df_combined.sample(frac=1).reset_index(drop=True)

# 4. Write combined data to labeledData.csv
df_shuffled.to_csv("labeledData.csv", columns=["text", "label"], index=False)

# 5. Split into train (80%), eval (10%), and test (10%) sets
total_count = len(df_shuffled)
train_end = int(0.8 * total_count)
eval_end = train_end + int(0.1 * total_count)

train_df = df_shuffled.iloc[:train_end]
eval_df = df_shuffled.iloc[train_end:eval_end]
test_df = df_shuffled.iloc[eval_end:]

# 6. Save splits to CSV
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Data successfully split into train.csv (80%), eval.csv (10%), and test.csv (10%).")
