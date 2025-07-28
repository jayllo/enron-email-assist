# IE7374
GenAI Coursework IE7374
---

To install the project, clone this repo and run docker compose up.
The docker will install all necessary requirements & start Jupyter notebook at http://localhost:8888.
We are in the process of transitioning off Notebooks to .py files.

---

Our project currently consists of notebooks placed in /src 
- Prprocessing 
  - 1_preprocessing_v7 which cleans the enron data
- Model training and evaluation files
  - data_prep_step1
  - data_prep_step2
  - data_prep_step3
  - train_step4
  - evaluate_step5
---

**Setup**

1. Right now, we have to manually download the EnronData set due to size.
Enron Dataset - https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

Update 7/26: We found the Enron dataset online and can now directly load from CMU: https://www.cs.cmu.edu/~enron/ 

3. Load it into drive/local
4. Run proprocessing
5. Run the data prep scripts in sequence (labeled step1-3)
6. Run training
7. Run evaluate

---

**Model Selection & Training**

We will build our email‐reply generator on GPT-2 Small (124 M parameters) because it delivers fluent, general-purpose text generation without requiring massive compute resources 
This is done in [train_step4](https://github.com/jayllo/enron-email-assist/blob/main/src/train_step4.ipynb)

During training, we will token-limit each prompt to 512 tokens, ensuring all formats remain comparable in length.
We will define a small, fixed set of tokens (e.g., [formal], [friendly], [urgent]) and insert exactly one at the start of every training example. During fine‐tuning, the model learns to associate each token with its respective tone. At inference time, users simply swap that token to generate the same core reply in different styles.
