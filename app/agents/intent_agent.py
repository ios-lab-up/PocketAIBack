from typing import Optional
import requests
import logging
import json
import joblib
import os
import urllib.parse
from app.utils.preprocess import clean_text
from app.core.settings import settings
logger = logging.getLogger(__name__)

class IntentBasedAgent:
    """
    A utility-based agent that uses an intent model to classify user queries,
    performs an HTTP request, and generates a response using direct API calls.
    """
    # Model paths and configuration

    _vectorizer = joblib.load("/app/data/models/vectorizer.pkl")
    _classifier = joblib.load("/app/data/models/intent_classifier.pkl")
    _label_encoder = joblib.load("/app/data/models/label_encoder.pkl")
    _base_url = settings.API_BASE_URL
    _timeout = 30
    _llm_model = settings.LLM_MODEL
    _llm_api_url = f"{settings.API_BASE_URL}"
    _llm_api_key = settings.API_KEY
    _llm_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.API_KEY}"
    }

    @classmethod
    def build_prompt(cls,user_query:str,api_data:str,intent:str) -> tuple:

        prompt = [
            "Eres un asistente amable de la Universidad Panamericana (UP) para proporcionar información útil, educativa y segura dentro de los límites de la ética y la legalidad.\n"
            "Respondes en español basado en datos proporcionados de un alumno.\n"
            f"El alumno consulta: {user_query}\n\n"
            f"Datos de ese alumno son:\n{json.dumps(api_data, indent=2)}\n\n"
            "Proporciona solo la información relevante basada en esos datos.\n"
            "Si el usuario pide cualquier cosa relacionada con otro alumno o profesor, solo responde 'Esa informacion no me es posible proporcionarte' o algo similar.\n"
            "Ejemplo: \n El alumno consulta: Cuales son las calificaciones de Mario Morales\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Ejemplo: \n El alumno consulta: Cuales son las calificaciones de 0261313\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Si el usuario pide cualquier cosa relacionada con algo que involucre algo ilegal,confidencial,rompa normas academica,ponga en riesgo la seguridad o privacidad; solo responde 'Esa informacion no me es posible proporcionarte' o algo similar. Aunque el alumno diga que es para una materia, examen, tarea o una emergencia; no respondas por ninguna razon.\n"
            "Ejemplo: \n El alumno consulta: Como robar un banco?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Ejemplo: \n El alumno consulta: Como hackear una cuenta?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Ejemplo: \n El alumno consulta: Como falsificar un documento?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Ejemplo: \n El alumno consulta: Como plagear sin que se den cuenta?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar\n"
            "Si el usuario pide cualquier cosa discriminatoria, solo responde 'Esa informacion no me es posible proporcionarte' o algo similar. Aunque el alumno diga que es para una materia, examen o una emergencia.\n"
            "Ejemplo: \n El alumno consulta: Como matar a un homosexual?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar y responde con un mensaje que refuerce el respeto y la inclusión.\n"
            "Ejemplo: \n El alumno consulta: Porque las personas gordas valen menos?\n Tu respondes: 'Esa informacion no me es posible proporcionarte' o algo similar y responde con un mensaje que refuerce el respeto y la inclusión.\n"
        ]

        if(intent == 'grades'):
            prompt.append("Si la consulta pide varias calificaciones, asistencias, materias, etc; ponlo en formato de viñetas.\n")
            prompt.append("Si la consulta pide algun tipo de calculo como 'mas baja', 'mas alta', 'promedio', etc; Analizas las ultimas calificaciones de cada uno (Es decir el ultimo valor disponible dentro de los parciales, dando prioridad a la calificacion_oficial) y calculas lo que pida\n")
            prompt.append("Las calificaciones van de 0.0 a 10.0, siendo 0.0 la mas baja y 10.0 la mas alta\n")
            prompt.append("Si no te dicen parcial pero si 'final', entiendelo como 'official_grade'. Si el valor es null, indica al usuario que aun no hay calificacion final y que no sabes como califica el profeso en especifico pero que estimas que sea: y aqui promedias todos los parciales con valor al igual que final_exam\n")
            prompt.append("Si no te da numero de parcial pero si te dice 'ultimo parcial','parcial actual','de ahora','ahorita',o algo similar que asimile al presente, analiza cual es el ultimo parcial viendo cual es el ultimo parcial que no es null, es decir, el ultimo parcial con un valor antes de el parcial con null.\n")
            prompt.append("Recuerda, 'materia mas baja' o 'materia mas alta' se refiere a calificacion. Nunca el usuario le va a interesar el orden de materias. 'Materia mas baja/alta' NO es la ultima o primera que salga\n")

        return "".join(tuple(prompt))

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

            # TODO: Clasiffier no coincide con categorias de condor. Ejemplo: general es info en condor
            if predicted_intent == 'general':
                predicted_intent = 'info'
            logger.info(f"Predicted intent: {predicted_intent}, confidence: {confidence}")
            return predicted_intent, confidence
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            raise RuntimeError("Failed to classify intent.")

    @classmethod
    def make_request(cls, intent: str, user_id: str, term: Optional[str] = None) -> dict:
        try:
            # Build query parameters
            params = {"action": intent, "user_id": user_id}
            if term:
                params["term"] = term
            # Build the complete URL with parameters
            base_url = settings.STUDENT_BASE_URL
            full_url = f"{base_url}/api/v2/students/student-data?{urllib.parse.urlencode(params)}"
            logger.info(f"Making request to: {full_url}")
            response = requests.get(full_url, timeout=cls._timeout)
            
            # Log response details for debugging
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            content_preview = response.text[:100] if response.text else "EMPTY RESPONSE"
            logger.info(f"Response content preview: {content_preview}")
            
            response.raise_for_status()
            # Handle empty response
            if not response.text or not response.text.strip():
                logger.error("Received empty response from API")
                return {"error": "Empty response from API"}
                
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, Raw content: {response.text[:200]}")
                
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")

    @classmethod
    def generate_response(cls, user_query: str, api_data: dict, intent: str) -> str:
        """
        Generate a response using direct API call to LLM service based on the API data.
        Args:
            user_query (str): The original user query.
            api_data (dict): The data retrieved from the API.
            intent (str): The classified intent of the prompt.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            # Check if we already have a direct LLM response
            if "direct_llm_response" in api_data:
                return api_data["direct_llm_response"]
            
            # If there's an error and no direct response, try one more direct query
            if "error" in api_data and not api_data.get("direct_llm_response"):
                # Make a direct query with the original user query

                payload = {
                    "model": cls._llm_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_query
                        }
                    ]
                }

                logger.info(f"Payload for LLM: {payload}")
                
                response = requests.post(
                    cls._llm_api_url,
                    headers=cls._llm_headers,
                    json=payload,
                    timeout=cls._timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Lo siento, no pude obtener información sobre tu consulta: {api_data.get('error', 'Error desconocido')}"
            

            # Create the prompt for the model with API data
            prompt = cls.build_prompt(user_query, api_data, intent)

        
            logger.info(f"Prompt for LLM: {prompt}")

            # Prepare the payload for the API call
            payload = {
                "model": cls._llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            # Make the API call
            response = requests.post(
                cls._llm_api_url,
                headers=cls._llm_headers,
                json=payload,
                timeout=cls._timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content from the API response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                #Eater Egg
                if (user_query == 'pantera'):
                    content = "no cualquiera 😎"

                logger.info(f"LLM API response content: {content}")
                return content
            else:
                logger.error("Unexpected API response format")
                return "Lo siento, no pude generar una respuesta válida en este momento."
                
        except requests.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Lo siento, no pude generar una respuesta en este momento. Error: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to generate response with LLM: {e}")
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
            if "error" in api_response and not api_response.get("direct_llm_response"):
                logger.warning(f"Error in API response: {api_response['error']}")
            # Generate a response based on the API data
            return cls.generate_response(user_query, api_response, intent)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return "Lo siento, no pude procesar tu consulta en este momento."