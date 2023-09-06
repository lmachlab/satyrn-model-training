import pandas as pd
import os

dfs = []
csv_directory = r'/Users/lauramachlab/Library/CloudStorage/OneDrive-Personal/Documents/_northwestern/_MSAI/c3 lab/satyrn/Evaluated Reports'  # Replace with the directory where your CSV files are stored
count = 0
unsuccessful_files = []
for filename in os.listdir(csv_directory):
    print(count)
    count += 1
    print(filename)
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'cp1252']

    # Initialize a DataFrame
    df = None

    if filename.endswith('.csv'):

        # Try reading the file with different encodings
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(os.path.join(csv_directory, filename), encoding=encoding)
                break  # If successful, break out of the loop
            except Exception as e:
                print(f"Error reading {filename} with encoding {encoding}: {e}")
        if df is not None:
            # Rename the problematic column if needed
            df.rename(columns={'# of claims': '# of claims '}, inplace=True)
            df.rename(columns={'# claims': '# of claims '}, inplace=True)
            df.rename(columns={'#of claims': '# of claims '}, inplace=True)
            df.rename(columns={'# of claim': '# of claims '}, inplace=True)

            # Extract the two columns you need
            df = df[['claims', '# of claims ']]

            # Append the DataFrame to the list
            dfs.append(df)
        else:
            unsuccessful_files.append(filename)
            print(f"Could not read {filename} with any encoding, skipping...")

    elif filename.endswith('.xlsx'):
        
        df = pd.read_excel(os.path.join(csv_directory, filename))
        # Try reading the file with different encodings
        # for encoding in encodings_to_try:
        #     try:
        #         df = pd.read_excel(os.path.join(csv_directory, filename), encoding=encoding)
        #         break  # If successful, break out of the loop
        #     except Exception as e:
        #         print(f"Error reading {filename} with encoding {encoding}: {e}")
        if df is not None:
            # Rename the problematic column if needed
            df.rename(columns={'# of claims': '# of claims '}, inplace=True)
            df.rename(columns={'# claims': '# of claims '}, inplace=True)
            df.rename(columns={'#of claims': '# of claims '}, inplace=True)
            df.rename(columns={'# of claim': '# of claims '}, inplace=True)

            # Extract the two columns you need
            df = df[['claims', '# of claims ']]

            # Append the DataFrame to the list
            dfs.append(df)
        else:
            unsuccessful_files.append(filename)
            print(f"Could not read {filename} with any encoding, skipping...")




# Save the consolidated DataFrame to a CSV file

consolidated_df = pd.concat(dfs, ignore_index=True)
consolidated_df.to_csv('consolidated_data.csv', index=False)
