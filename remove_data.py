import pandas as pd

dataset_path = r'/Users/lauramachlab/Library/CloudStorage/OneDrive-Personal/Documents/_northwestern/_MSAI/c3 lab/satyrn/cleaned_data.csv'

df = pd.read_csv(dataset_path)
condition = df['# of claims '] <= 6

# Use the boolean mask to drop the rows
filtered_df = df[condition]

filtered_df.to_csv('filtered_data_6.csv', index=False)