# IE7374
GenAI Coursework IE7374
---

Our project currently consists of a GPT2 trained and tuned with LORA.
The launchpoint for this is from sample_model_runner.
---

**Setup**

1. For demo, you can simply clone the repo 
   git clone
2. docker compose up model-runner
   - which will run the sample_model_runner against a sample data set.

To use the full Enron Dataset - https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

You have to
1. git clone our repo
2. download the Enron email data set
3. Replace data/sample_raw_dataset.csv with the enron email dataset
4. Run sample_model_runner
---

**Model Selection & Training**

We will build our email‐reply generator on GPT-2 Small (124 M parameters) because it delivers fluent, general-purpose text generation without requiring massive compute resources 
This is done in [train_step4](https://github.com/jayllo/enron-email-assist/blob/main/src/train_step4.ipynb)

During training, we will token-limit each prompt to 512 tokens, ensuring all formats remain comparable in length.
We will define a small, fixed set of tokens (e.g., [formal], [friendly], [urgent]) and insert exactly one at the start of every training example. During fine‐tuning, the model learns to associate each token with its respective tone. At inference time, users simply swap that token to generate the same core reply in different styles.
