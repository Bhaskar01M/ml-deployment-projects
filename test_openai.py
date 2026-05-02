import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file
load_dotenv()

# Check if API key is set
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ API Key found!")
    masked_key = api_key[:7] + "..." + api_key[-4:]
    print(f"   Key: {masked_key}")
else:
    print("❌ API Key not found in .env file")
    exit()

# Try to initialize OpenAI client
try:
    client = OpenAI()
    print("✅ OpenAI client initialized successfully!")
    print("\n🎉 YOU'RE READY TO BUILD THE RAG SYSTEM!")
except Exception as e:
    print(f"❌ Error: {e}")