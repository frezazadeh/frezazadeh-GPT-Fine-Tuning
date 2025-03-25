![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT4o-orange.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)
![Claude](https://img.shields.io/badge/Claude-%203.5-yellow)


# 🧠 GPT-Fine-Tuning
This repository contains a fine-tuning project for building a model that estimates product prices from product descriptions. If you also need my train dataset, please contact me (farhad.rh@ieee.org)
---
## 📂 Project Structure
```plaintext
frezazadeh-GPT-Fine-Tuning/
├── data/
│   ├── processed/            # Preprocessed datasets (train.pkl, test.pkl)
├── src/                      # Core source code
│   ├── __init__.py
│   ├── config.py             # Environment variables and API configuration
│   ├── fine_tune.py          # Fine-tuning process
│   ├── items.py              # Item class for processing product descriptions
│   ├── testing.py            # Unit testing framework
│   ├── utils.py              # Utility functions
├── .env                      # API keys and environment variables
├── README.md                 # Project documentation (this file)
├── main.py                   # Main entry point for execution
└── items.py                  # Alias to fix unpickling issues
```



## 📌 Features
- ✅ Fine-tunes **GPT-4o** for product price estimation  
- ✅ Uses **Hugging Face Transformers** for tokenization  
- ✅ Organizes data into structured **JSONL** format  
- ✅ Monitors fine-tuning with **Weights & Biases**  
- ✅ Implements **unit testing** for robust validation  
- ✅ Fully modular and professional **Python package**  

## Set Up API Keys

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

## Run Fine-Tuning

```sh
python main.py
```

## 📚 Reference

[edwarddonner](https://edwarddonner.com/)
