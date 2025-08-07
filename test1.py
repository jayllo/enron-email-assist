# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 19:17:57 2025

@author: annas
"""

from multiprocessing import Pool
import pandas as pd

from preprocessing_pipeline_v2 import (
    split_metadata_and_body,
    final_scrub_text,
    format_and_split_thread,
    anonymize_names
)

import re
from sklearn.model_selection import train_test_split
import spacy

from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

# Function to extract Subject from metadata block
def extract_subject(metadata_block):
    match = re.search(r'Subject:\s*(.*)', metadata_block, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Main processing function
def process_email_row(row):
    metadata_block, message_body = row['metadata_block'], row['message_body']
    message_id_match = re.search(r'Message-ID:\s*<(.*?)>', metadata_block, flags=re.IGNORECASE)
    message_id = message_id_match.group(1) if message_id_match else "unknown"

    subject = extract_subject(metadata_block)
    threaded_messages = format_and_split_thread(message_body)

    entries = []
    for part in threaded_messages:
        clean_text = anonymize_names(part['text'])  # Apply spaCy-based name anonymization
        entries.append({
            'email_id': message_id,
            'tag': part['tag'],
            'subject': subject,
            'email_text': clean_text
        })

    return entries

def compute_upper_bound(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q3 + multiplier * iqr


if __name__ == '__main__':
    # 1. Load data
    df = pd.read_csv("sample_raw_dataset.csv")

    # 2. Split metadata and message body
    df[['metadata_block', 'message_body']] = df['message'].apply(
        lambda x: pd.Series(split_metadata_and_body(x))
    )

    # 3. Filter outliers based on message body length
    df['message_length'] = df['message_body'].apply(len)
    upper_limit = compute_upper_bound(df['message_length'])
    df = df[df['message_length'] <= upper_limit].copy()

    # 4. Process emails in parallel
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_email_row, df.to_dict('records')), total=len(df)))

    
    # 5. Flatten and convert to DataFrame
    flattened = [entry for sublist in results for entry in sublist]
    df_final = pd.DataFrame(flattened)

    # 6. Shuffle & split
    train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 7. Save
    train_df.to_csv("enron_cleaned_train.csv", index=False)
    val_df.to_csv("enron_cleaned_val.csv", index=False)
    test_df.to_csv("enron_cleaned_test.csv", index=False)

    print("✅ Done. Saved cleaned files with columns: email_id, tag, subject, email_text.")
