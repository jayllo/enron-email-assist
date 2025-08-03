import pandas as pd
from preprocessing_pipeline import preprocess_pipeline 


# Load your raw data
df = pd.read_csv("sample_raw_dataset.csv")
# df = pd.read_csv("emails.csv")

# Run your pipeline
df_clean = preprocess_pipeline(df)

# Save or continue
df_clean.to_csv("clean_data.csv", index=False)
