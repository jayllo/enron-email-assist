import pandas as pd
import numpy as np
import re
from multiprocessing import Pool
import spacy
from sklearn.model_selection import train_test_split

# ------------------------
# 1. Load Data
# ------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset='message')
    return df

# ------------------------
# 2. Split Metadata & Body
# ------------------------
def split_metadata_and_body(email_text):
    pattern = r'(Message-ID:.*?X-FileName:.*?\n)'
    match = re.search(pattern, email_text, re.DOTALL | re.IGNORECASE)
    if match:
        metadata = match.group(1).strip()
        body = email_text[match.end():].strip()
    else:
        metadata, body = '', email_text.strip()
    return metadata, body

def apply_metadata_split(df):
    df[['metadata_block', 'message_body']] = df['message'].apply(
        lambda x: pd.Series(split_metadata_and_body(x))
    )
    return df

# ------------------------
# 3. Filter by Length
# ------------------------
def compute_upper_bound(series, multiplier=1.5):
    """Compute the dynamic upper bound for filtering using IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q3 + multiplier * iqr

def filter_by_length(df, text_col="text", multiplier=1.5):
    """Filter out rows with unusually long text values."""
    df = df.copy()
    df["length"] = df[text_col].apply(len)
    upper_limit = compute_upper_bound(df["length"], multiplier)
    return df[df["length"] <= upper_limit].drop(columns=["length"])


# ------------------------
# 4. Scrubbing Utilities
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

    text = re.sub(r'\b[\w\.-+=_%]+@[\w\.-]+\.\w{2,}\b', '<ANON_EMAIL>', text)
    text = re.sub(r'(\b(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b)', '<ANON_PHONE>', text)
    text = re.sub(r'\b[A-Z][a-z]+,\s[A-Z][a-z]+\b', '<ANON_NAME>', text)
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '<ANON_NAME>', text)
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

def process_email_row(row):
    message_id = row['metadata_block'].split('\n')[0]
    body_text = row['message_body']
    threaded_messages = format_and_split_thread(body_text)
    return [
        {'message_id': message_id, 'tag': msg['tag'], 'clean_message': msg['text']}
        for msg in threaded_messages
    ]

# ------------------------
# 6. Anonymization
# ------------------------
nlp = spacy.load("en_core_web_sm")

def anonymize_names(text):
    doc = nlp(text)
    return " ".join("[NAME]" if token.ent_type_ == "PERSON" else token.text for token in doc)

# ------------------------
# 7. Main Execution
# ------------------------
def main():
    df = load_dataset('emails.csv') #Enron dataset from Kaggle
    df = apply_metadata_split(df)
    df = filter_by_length(df)

    with Pool() as pool:
        results = pool.map(process_email_row, df.to_dict('records'))
    df_clean = pd.DataFrame([item for sublist in results for item in sublist])

    df_clean['clean_message'] = df_clean['clean_message'].apply(anonymize_names)
    df_final = df_clean.rename(columns={'clean_message': 'anon_message'})

    train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("enron_cleaned_v7.csv", index=False)
    val_df.to_csv("enron_val_v7.csv", index=False)
    test_df.to_csv("enron_test_v7.csv", index=False)

if __name__ == '__main__':
    main()
