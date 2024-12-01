import os
import pandas as pd
import re
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================
# Configuration and Setup
# =============================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("intent_dataset_generation.log"),
    ],
)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    logger.error("OPENAI_API_KEY is not set in the environment variables.")
    exit(1)

# =============================
# HTTP Session with Retry Strategy
# =============================

def create_session():
    """Creates a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

# Initialize the session
session = create_session()

# =============================
# API Interaction Functions
# =============================

def get_chatgpt_response(prompt):
    """Sends a prompt to OpenAI API and retrieves the response."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = session.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        else:
            logger.error("Unexpected API response structure.")
            return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return ""

# =============================
# Question Cleaning Functions
# =============================

def clean_questions(raw_text):
    """Cleans the raw text into a list of questions."""
    raw_questions = raw_text.strip().split("\n")
    clean_questions = []
    for question in raw_questions:
        question = re.sub(r"^\d+\.\s*", "", question).strip()  # Remove numbering
        if question:
            clean_questions.append(question)
    return clean_questions

def generate_questions_for_intent(prompt):
    """Generates questions for a specific intent using the given prompt."""
    response = get_chatgpt_response(prompt)
    if not response:
        return []
    return clean_questions(response)

# =============================
# Dataset Generation
# =============================

def generate_intent_dataset(intents, examples_per_intent):
    """
    Generates a dataset for intent prediction based on predefined intents.
    Args:
        intents (dict): A dictionary with intent names as keys and prompts as values.
        examples_per_intent (int): Number of examples to generate per intent.
    Returns:
        pd.DataFrame: A DataFrame containing the dataset.
    """
    dataset = []
    for intent, prompt in intents.items():
        logger.info(f"Generating questions for intent: {intent}")
        questions = []
        for _ in tqdm(range(examples_per_intent // 10), desc=f"Intent: {intent}"):
            batch = generate_questions_for_intent(prompt)
            questions.extend(batch)
        dataset.extend([{"query": q, "intent": intent} for q in questions[:examples_per_intent]])
    return pd.DataFrame(dataset)

# =============================
# Main Script
# =============================

def main():
    # Define intents and their prompts
    intents = {
        "grades": (
            "Genera una lista de preguntas en español relacionadas con calificaciones, "
            "resultados de exámenes o desempeño académico. Ejemplos: "
            "¿Cuánto necesito para pasar matemáticas? ¿Cómo puedo mejorar mis calificaciones?"
        ),
        "attendance": (
            "Genera una lista de preguntas en español relacionadas con asistencias, faltas o tardanzas. "
            "Ejemplos: ¿Cuántas faltas tengo en historia? ¿Cuántas faltas me quedan antes de reprobar?"
        ),
        "schedule": (
            "Genera una lista de preguntas en español relacionadas con horarios de clases. "
            "Ejemplos: ¿A qué hora empieza mi clase de física? ¿Cuántas horas de clase tengo al día?"
        ),
        "links": (
            "Genera una lista de preguntas en español relacionadas con links importantes o recursos académicos. "
            "Ejemplos: ¿Dónde está el link al portal del estudiante? ¿Cuál es la wiki de ingeniería?"
        ),
        "general": (
            "Genera una lista de preguntas generales en español que un estudiante podría hacer. "
            "Ejemplos: ¿Cuál es la cafetería más cercana? ¿Qué eventos hay esta semana?"
        ),
    }

    # Number of examples per intent
    examples_per_intent = 1000

    # Generate the dataset
    logger.info("Starting dataset generation...")
    dataset = generate_intent_dataset(intents, examples_per_intent)

    # Save to CSV
    output_file = "intent_dataset.csv"
    dataset.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()
