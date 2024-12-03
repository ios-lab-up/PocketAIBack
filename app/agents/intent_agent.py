from typing import Optional
from langchain_openai import ChatOpenAI
import requests
import logging
import json
import joblib
import urllib.parse
from app.utils.preprocess import clean_text
from app.core.settings import settings

logger = logging.getLogger(__name__)

class IntentBasedAgent:
    """
    A utility-based agent that uses an intent model to classify user queries,
    performs an HTTP request, and generates a response using ChatGPT.
    """

    # Carga de modelos como variables de clase
    _vectorizer = joblib.load("app/models/vectorizer.pkl")
    _classifier = joblib.load("app/models/intent_classifier.pkl")
    _label_encoder = joblib.load("app/models/label_encoder.pkl")
    _llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    _base_url =  settings.API_BASE_URL # Base URL del endpoint
    _timeout = 30

    @classmethod
    def classify_intent(cls, message: str):
        """
        Classify the intent of a message using the intent classifier.

        Args:
            message (str): The user query.

        Returns:
            tuple: (predicted_intent, confidence)
        """
        try:
            clean_query = clean_text(message)
            query_vectorized = cls._vectorizer.transform([clean_query])
            predicted_label = cls._classifier.predict(query_vectorized)[0]
            predicted_intent = cls._label_encoder.inverse_transform([predicted_label])[0]
            probabilities = cls._classifier.predict_proba(query_vectorized)[0]
            confidence = max(probabilities)

            logger.info(f"Predicted intent: {predicted_intent}, confidence: {confidence}")
            return predicted_intent, confidence
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            raise RuntimeError("Failed to classify intent.")


    @classmethod
    def make_request(cls, intent: str, user_id: str, term: Optional[str] = None) -> dict:
        """
        Perform an HTTP GET request to the `/student-data` endpoint.

        Args:
            intent (str): The intent derived from the user's query.
            user_id (str): The user identifier.
            term (str, optional): Optional term for term-specific actions.

        Returns:
            dict: The JSON response from the API or an error message.
        """
        try:
            # Construir los parámetros de consulta
            params = {"action": intent, "user_id": user_id}
            if term:
                params["term"] = term

            # Construir la URL completa con parámetros
            base_url = f"{cls._base_url}/student-data"
            full_url = f"{base_url}?{urllib.parse.urlencode(params)}"

            logger.info(f"Making request to: {full_url}")  # Log con la URL completa

            response = requests.get(full_url, timeout=cls._timeout)
            response.raise_for_status()

            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
    @classmethod
    def generate_response(cls, user_query: str, api_data: dict) -> str:
        """
        Generate a response using ChatGPT based on the API data.

        Args:
            user_query (str): The original user query.
            api_data (dict): The data retrieved from the API.

        Returns:
            str: The generated response from ChatGPT.
        """
        try:
            # Crear el prompt para el modelo
            prompt = (
                "Eres un asistente que responde en español basado en datos de una API.\n"
                f"Consulta del usuario: {user_query}\n\n"
                f"Datos de la API:\n{json.dumps(api_data, indent=2)}\n\n"
                "Genera una respuesta clara y útil para el usuario basada en estos datos."
            )
            logger.info(f"Prompt for ChatGPT: {prompt}")

            # Usar invoke para obtener la respuesta
            response = cls._llm.invoke(prompt)

            # Acceder al contenido del mensaje (AIMessage.content)
            if hasattr(response, "content"):
                return response.content.strip()

            # Si el formato inesperado ocurre, registrar un error
            logger.error(f"Unexpected response type from ChatGPT: {response}")
            return "Lo siento, no pude generar una respuesta válida en este momento."
        except Exception as e:
            logger.error(f"Failed to generate response with ChatGPT: {e}")
            return "Lo siento, no pude generar una respuesta en este momento."



    @classmethod
    def handle_query(cls, user_query: str, user_id: str, term: Optional[str] = None) -> str:
        """
        Handle a user query by classifying the intent, making an API request, and generating a response.

        Args:
            user_query (str): The user's query.
            user_id (str): The user identifier.
            term (str, optional): Optional term for term-specific actions.

        Returns:
            str: The generated response for the user.
        """
        try:
            # Classify the intent of the user's query
            intent, confidence = cls.classify_intent(user_query)

            # Perform the API request
            api_response = cls.make_request(intent, user_id, term)
            if "error" in api_response:
                return f"Error al realizar la consulta: {api_response['error']}"

            # Generate a response based on the API data
            return cls.generate_response(user_query, api_response)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return "Lo siento, no pude procesar tu consulta en este momento."
