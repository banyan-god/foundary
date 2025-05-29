import pandas as pd

input_csv = "../data/transactions.csv"      # Path to your CSV file
output_txt = "output.txt"    # Path for the output text file


df = pd.read_csv(input_csv)

with open(output_txt, "w", encoding="utf-8") as f:
    for col in df.columns:
        for value in df[col].dropna():
            f.write(str(value).replace('\n', ' ').replace('\r', ' ') + '\n')