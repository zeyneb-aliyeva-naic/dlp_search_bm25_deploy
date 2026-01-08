import os
import logging
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# Elasticsearch Configuration
ES_URL =  "http://10.3.3.16:9200" # "http://10.3.3.16:9200"
ES_API_KEY = "" # "" 
HOST = "https://51757c4bec8a4582904feabde61b08c0.us-central1.gcp.cloud.es.io:443"
API_KEY = "c1dBWmFwc0I0SUNqY2JWczEyRXo6YkJjUEtFY25Qdmh0b0ktZ2R6cTlwdw=="
ES_INDEX = "flattened_hscodes_v6"
ES_INDEX_EN = "final_flattened_hscodes_en"
ES_INDEX_RU = "final_flattened_hscodes_ru_v2"

SOURCE_ES_URL = "http://10.3.3.16:9200"
SOURCE_ES_API_KEY = ""
ORGANIZATIONS_INDEX = "organizations_v3"

SPELLING_MODEL_DIR = os.getenv("SPELLING_MODEL_DIR", "spelling_model_deployment.pkl")


from sentence_transformers import SentenceTransformer
import os
import logging

logger = logging.getLogger(__name__)

def load_model():
    """
    Load a SentenceTransformer model from HF Hub using token.
    Returns the model if successful, otherwise None.
    """
    model_name = os.getenv("MODEL_REPO")
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # set this in your environment

    try:
        model = SentenceTransformer(model_name, use_auth_token=hf_token)
        emb_dim = model.get_sentence_embedding_dimension()
        logger.info(f"✅ Model '{model_name}' loaded successfully with embedding dim {emb_dim}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model '{model_name}': {e}", exc_info=True)
        return None


def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
