import os
import logging
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer

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

# Model Configuration
DEFAULT_LOCAL_DIR = ""
DEFAULT_HF_REPO = "DmitriyKuramshin/m12_1e"
HF_TOKEN_ENV_VAR = "HUGGINGFACE_HUB_TOKEN"

#MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_LOCAL_DIR)
MODEL_REPO = os.getenv("MODEL_REPO", DEFAULT_HF_REPO)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "m12_1e"

SPELLING_MODEL_DIR = os.getenv("SPELLING_MODEL_DIR", "spelling_model_deployment.pkl")


def load_model() -> Optional[SentenceTransformer]:
    """
    Load the sentence transformer model from local dir if it exists,
    otherwise from the Hugging Face repo.

    Returns:
        SentenceTransformer model or None if loading fails
    """
    logger.info(f"Loading model. MODEL_DIR={MODEL_DIR}, MODEL_REPO={MODEL_REPO}")

    try:
        if Path(MODEL_DIR).is_dir():
            logger.info(f"Found local model directory at '{MODEL_DIR}', loading from disk.")
            model = SentenceTransformer(str(MODEL_DIR))

        else:
            logger.info(
                f"Local directory '{MODEL_DIR}' not found. "
                f"Loading from Hugging Face repo '{MODEL_REPO}'."
            )

            hf_token = os.getenv(HF_TOKEN_ENV_VAR)

            if hf_token:
                logger.info(f"Using Hugging Face token from env var '{HF_TOKEN_ENV_VAR}'.")
                # use_auth_token works with private repos
                model = SentenceTransformer(str(MODEL_REPO), use_auth_token=hf_token)
            else:
                logger.warning(
                    f"Env var '{HF_TOKEN_ENV_VAR}' is not set. "
                    "Trying to load without a token. This will fail if the repo is private."
                )
                model = SentenceTransformer(MODEL_REPO)

        dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {dim}")
        return model

    except Exception as e:
        logger.warning(
            f"Warning. Could not load model from '{MODEL_DIR}' or '{MODEL_REPO}': {e}"
        )
        logger.warning("Vector search will be disabled.")
        return None


def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
