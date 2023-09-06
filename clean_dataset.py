import pandas as pd

dataset_path = r'/Users/lauramachlab/Library/CloudStorage/OneDrive-Personal/Documents/_northwestern/_MSAI/c3 lab/satyrn/consolidated_data.csv'

df = pd.read_csv(dataset_path)
df_filtered = df.dropna(subset=['claims', '# of claims '], how='any')
df_filtered.to_csv('cleaned_data.csv', index=False)