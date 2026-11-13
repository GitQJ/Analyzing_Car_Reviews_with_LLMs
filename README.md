# Analyzing_Car_Reviews_with_LLMs
Used LLMs to solve diverse language tasks for a car dealership company

## Car Review Analysis with Hugging Face Transformers: Sentiment, Translation, QA & Summarization | Python, NLP

Used four pre-trained Hugging Face models to analyse car reviews for a dealership scenario: sentiment classification, English-to-Spanish translation, extractive question answering and summarisation. Each task was scored with an appropriate metric where a reference was available.

# CONTEXT
A car sales and rental company wanted to prototype language features that help customers and support agents. The pilot covered classifying reviews, serving Spanish-speaking customers, answering questions about a review and condensing long feedback.

# MY ROLE
I wrote the inference code and the evaluation for all four tasks.

# WHAT I DID
• Sentiment: ran distilbert-base-uncased-finetuned-sst-2-english on the 5 reviews in car_reviews.csv, mapped labels to 0/1 and computed accuracy and F1 with the evaluate library
• Translation: translated review text into Spanish with Helsinki-NLP/opus-mt-en-es and scored it with BLEU against provided reference translations
• Question answering: loaded deepset/minilm-uncased-squad2 through AutoModelForQuestionAnswering and decoded the answer span from the start and end logits, rather than using a ready-made pipeline
• Summarisation: condensed the last review to about 53 tokens with cnicu/t5-small-booksum

# RESULTS
• Sentiment: accuracy 0.8, F1 0.86 on 5 reviews
• Translation: BLEU 0.6 on 2 sentence
• QA answer to "What did he like about the brand?": "ride quality, reliability"
• Summary: "'the Nissan Rogue provides me with the desired SUV experience without burdening me with an exorbitant payment; the financial arrangement is quite reasonable."

# TECH STACK
Python, Hugging Face Transformers, PyTorch, evaluate (accuracy, F1, BLEU), pandas
