import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
HF_TOKEN = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')

