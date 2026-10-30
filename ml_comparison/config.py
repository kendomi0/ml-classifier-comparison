import os
from dotenv import load_dotenv

load_dotenv(override=True)
ML_COMPARISON_DEBUG = os.getenv("ML_COMPARISON_DEBUG", "").lower() == "true"
