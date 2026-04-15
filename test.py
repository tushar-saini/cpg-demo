from evaluation.eval import run_eval
from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env

api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    run_eval()