# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:48:09 2025

@author: annas
"""



import pandas as pd
import numpy as np
import re
from multiprocessing import Pool
import spacy
from sklearn.model_selection import train_test_split

# ------------------------
# 1. Load Dataset
# ------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset='message')
    return df

# ------------------------
# 2. Metadata & Body Split
# ------------------------

def format_and_split_thread(body_text):
    # Match reply header blocks: To, Cc, Subject (optionally preceded by a timestamp/name)
    header_pattern = r'(?:\n\s{2,}To\s*:.*?\n\s{2,}cc\s*:.*?\n\s{2,}Subject\s*:.*?\n)'
    matches = list(re.finditer(header_pattern, body_text, flags=re.IGNORECASE))

    split_indices = [m.start() for m in matches]
    blocks = []

    # Add initial block if it exists before first header
    if split_indices and split_indices[0] > 0:
        blocks.append(body_text[:split_indices[0]].strip())

    for i in range(len(split_indices)):
        start = split_indices[i]
        end = split_indices[i+1] if i+1 < len(body_text) else len(body_text)
        block = body_text[start:end].strip()

        # Remove the header block from the body
        block = re.sub(header_pattern, '', block, flags=re.IGNORECASE).strip()
        blocks.append(block)

    # If no matches, treat entire body as one block
    if not blocks:
        blocks = [body_text.strip()]

    tagged_messages = []
    for i, block in enumerate(blocks):
        tag = '<|original|>' if i == len(blocks)-1 else f'<|reply{len(blocks)-1-i}|>'
        clean_text = final_scrub_text(block)
        if clean_text:
            tagged_messages.append({'tag': tag, 'text': clean_text})
    return tagged_messages



def split_metadata_and_body(email_text):
    pattern = r'(Message-ID:.*?X-FileName:.*?\n)'


    match = re.search(pattern, email_text, re.DOTALL | re.IGNORECASE)
    if match:
        metadata = match.group(1).strip()
        body = email_text[match.end():].strip()
    else:
        metadata, body = '', email_text.strip()
    return metadata, body

def extract_subject(metadata):
    match = re.search(r'Subject:\s*(.*)', metadata, re.IGNORECASE)
    return match.group(1).strip() if match else ''

def apply_metadata_split(df):
    df[['metadata_block', 'message_body']] = df['message'].apply(
        lambda x: pd.Series(split_metadata_and_body(x))
    )
    return df

# ------------------------
# 3. Length Filtering
# ------------------------
def compute_upper_bound(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q3 + multiplier * iqr

def filter_by_length(df, text_col="message_body", multiplier=1.5):
    df = df.copy()
    df["length"] = df[text_col].apply(len)
    upper_limit = compute_upper_bound(df["length"], multiplier)
    return df[df["length"] <= upper_limit].drop(columns=["length"])

# ------------------------
# 4. Text Scrubbing
# ------------------------
def final_scrub_text(text):
    lines = text.split('\n')
    first_real_line_index = 0
    for i, line in enumerate(lines):
        if re.match(r'^\s*(from|sent|to|cc|subject|date|forwarded):', line, re.IGNORECASE):
            continue
        if line.strip() not in ('', '>'):
            first_real_line_index = i
            break
    text = '\n'.join(lines[first_real_line_index:])

    disclaimer_patterns = [
        r'\*+\s*original message\s*\*+',
        r'this e-mail is the property of enron corp\..*',
        r'the information contained.*?designated recipients.*',
        r'internet communications are not secure.*'
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)


    text = re.sub(r'\b[\w.\-+=_%]+@[\w.\-]+\.\w{2,}\b', '<ANON_EMAIL>', text)
    text = re.sub(r'(\b(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b)', '<ANON_PHONE>', text)

    text = re.sub(r'\b[A-Z][a-z]+,\s[A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NAME]', text)
    text = re.sub(r'\bENRON@|enron@Enron\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', '<ANON_URL>', text)
    text = re.sub(r'^\s*>\s?.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'=[0-9A-F]{2}', '', text)
    text = re.sub(r'[-_*=]{3,}', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()



# ------------------------
# 5. Thread Formatting
# ------------------------
def format_and_split_thread(body_text):
    blocks = re.split(r'-{5,}\s*Original Message\s*-{5,}', body_text, flags=re.IGNORECASE)
    blocks = [block.strip() for block in blocks if block.strip()]
    tagged_messages = []
    for i, block in enumerate(blocks):
        tag = '<|original|>' if i == len(blocks)-1 else f'<|reply{len(blocks)-1-i}|>'
        clean_text = final_scrub_text(block)
        if clean_text:
            tagged_messages.append({'tag': tag, 'text': clean_text})
    return tagged_messages

# ------------------------
# 6. Anonymization
# ------------------------
nlp = spacy.load("en_core_web_sm")

def anonymize_names(text):
    doc = nlp(text)
    return " ".join("[NAME]" if token.ent_type_ == "PERSON" else token.text for token in doc)

# ------------------------
# 7. Row Processing
# ------------------------
def process_email_row(row):
    metadata = row['metadata_block']
    message_id = metadata.split('\n')[0]
    subject = extract_subject(metadata)
    body_text = row['message_body']
    threaded_messages = format_and_split_thread(body_text)
    return [
        {
            'email_id': message_id,
            'tag': msg['tag'],
            'subject': subject,
            'email_text': anonymize_names(msg['text'])
        }
        for msg in threaded_messages
    ]

# ------------------------
# 8. Main Pipeline Function
# ------------------------
def preprocess_pipeline_v2(input_csv_path, output_train_path="data/v7_sample_train.csv"):
    """
    Complete preprocessing pipeline for Enron email data.
    
    Args:
        input_csv_path (str): Path to the raw email CSV file
        output_train_path (str): Path to save the processed training data
    
    Returns:
        str: Path to the processed CSV file
    """
    df = load_dataset(input_csv_path)
    df = apply_metadata_split(df)
    df = filter_by_length(df)
    
    # Process emails - use single-threaded to avoid multiprocessing issues
    results = []
    for row in df.to_dict('records'):
        results.append(process_email_row(row))
    
    df_clean = pd.DataFrame([item for sublist in results for item in sublist])
    df_final = df_clean[['email_id', 'tag', 'subject', 'email_text']]
    df_final.to_csv(output_train_path, index=False)
    
    return output_train_path

if __name__ == '__main__':
    preprocess_pipeline_v2('data/sample_raw_dataset.csv', 'data/v7_sample_train.csv')
