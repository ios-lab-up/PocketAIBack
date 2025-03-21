from app.core.settings import settings  # Importar la instancia de configuración

from langchain.agents import Tool  # Tools for creating ReAct agents
from langchain.prompts import PromptTemplate  # For structured prompt creation
from app.utils.preprocess import clean_text  # Custom text preprocessing utility

import joblib  # For loading saved ML models
import requests  # For making HTTP requests to the student data API
import logging  # For application logging
import json  # For JSON parsing and handling
from pydantic import BaseModel, Field  # For data validation and schema definition

API_BASE_URL = settings.API_BASE_URL
API_KEY = settings.API_KEY

# Configure logger for this module
logger = logging.getLogger(__name__)

# Define the structure of agent responses using Pydantic
class AgentResponse(BaseModel):
    action: str = Field(description="El nombre de la acción a realizar.")  # The action to perform
    response: str = Field(description="La respuesta que se devolverá al usuario.")  # The response to return to the user

class StudentAgent:
    """
    Agent for student queries that integrates intent classification,
    data retrieval, and response generation using direct API calls.
    """

    def __init__(self, vectorizer_path="../data/models/vectorizer.pkl",
                 classifier_path="../data/models/intent_classifier.pkl",
                 label_encoder_path="../data/models/label_encoder.pkl",
                 llm_model="llama3.2:3b",
                 base_url=None, timeout=10,
                 llm_api_url=None,
                 llm_api_key=None):
        """
        Initialize the StudentAgent, including models and direct API setup.

        Args:
            vectorizer_path (str): Path to the vectorizer pickle file.
            classifier_path (str): Path to the classifier pickle file.
            label_encoder_path (str): Path to the label encoder pickle file.
            llm_model (str): Language model to use (default: "llama3.2:3b").
            base_url (str): Base URL for the student data API.
            timeout (int): Timeout for the student data API requests.
            llm_api_url (str): URL for the LLM API endpoint. Falls back to API_BASE_URL if None.
            llm_api_key (str): API key for authentication with the LLM API.
        """
        # Store configuration parameters
        self.llm_model = llm_model
        self.base_url = base_url
        self.timeout = timeout
        
        # Fall back to API_BASE_URL if llm_api_url is not provided
        self.llm_api_url = llm_api_url if llm_api_url else f"{API_BASE_URL}/api/chat/completions"
        self.llm_api_key = llm_api_key if llm_api_key else f"{API_KEY}"
        self.llm_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_api_key}"
        }
        
        logger.info(f"Initialized with LLM API URL: {self.llm_api_url}")
        
        # Define format instructions manually instead
        self.format_instructions = """
        {
          "action": "string", // El nombre de la acción a realizar
          "response": "string" // La respuesta que se devolverá al usuario
        }
        """

        # Load ML models for intent classification
        self._load_models(vectorizer_path, classifier_path, label_encoder_path)

        # Define available tools for use in prompts
        self.tools = self._define_tools()

    def _load_models(self, vectorizer_path, classifier_path, label_encoder_path):
        """
        Load the pre-trained ML models for intent classification from disk.
        """
        try:
            # Load vectorizer for text feature extraction
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load classifier model for intent prediction
            self.classifier = joblib.load(classifier_path)
            
            # Load label encoder to convert between numeric and text labels
            self.label_encoder = joblib.load(label_encoder_path)
            
            logger.info("Intent classification models loaded successfully.")
        except Exception as e:
            # Log and re-raise failure to load models as it's critical
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError("Failed to load intent classification models.")

    def fetch_student_data(self, action: str, user_id: str, term: str = None) -> dict:
        """
        Fetch student data from the API based on the specified action and parameters.
        
        Args:
            action (str): The type of data to fetch (e.g., "grades", "schedule")
            user_id (str): The ID of the student
            term (str, optional): Academic term specification
            
        Returns:
            dict: The fetched student data or error information
        """
        try:
            # Prepare request parameters
            params = {"action": action, "user_id": user_id}
            if term:
                params["term"] = term

            # Make API request to student data endpoint
            response = requests.get(
                f"{self.base_url}/student-data",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.RequestException as e:
            # Handle API request failures
            logger.error(f"Failed to fetch student data: {e}")
            return {"error": f"Failed to fetch student data: {str(e)}"}
    
    # Direct API call to llm
    def _call_llm_api(self, prompt):
        """
        Make a direct API call to the LLM service
        """
        try:
            # Generate tool names string
            tool_names = ", ".join([tool.name for tool in self.tools])
            
            # Format tools description in the same format that LangChain would use
            tools_description = self._format_tools_description()
            
            # Construct the system prompt using exact same format as the original template
            system_prompt = (
                "Eres un asistente útil que siempre responde en JSON. Responde a la siguiente entrada en este formato estricto:\n"
                f"{self.format_instructions}\n\n"
                "Herramientas disponibles:\n"
                f"{tools_description}\n\n"
                "Nombres de herramientas:\n"
                f"{tool_names}\n\n"
                "Acciones realizadas hasta ahora:\n"
                "[No hay acciones previas]\n\n"
                "Entrada del usuario:\n"
                f"{prompt}\n"
            )
            
            payload = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
            }
            
            response = requests.post(
                self.llm_api_url,
                headers=self.llm_headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content from the API response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.info(f"LLM API response content: {content}")
                return content
            else:
                logger.error("Unexpected API response format")
                return {"error": "Unexpected API response format"}
            
        except requests.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            return {"error": f"Error calling LLM API: {str(e)}"}
    
    # Helper method for formating
    def _format_tools_description(self):
        """
        Format the tools descriptions for inclusion in the system prompt.
        
        Returns:
            str: Formatted tools descriptions
        """
        tools_description = ""
        for tool in self.tools:
            tools_description += f"- {tool.name}: {tool.description}\n"
        return tools_description

    def _define_tools(self) -> list:
        """
        Define the tools available to the agent.
        
        Returns:
            list: List of Tool objects the agent can use
        """
        # Inner function to handle data fetching that will be used as a tool
        def fetch_data(inputs: dict) -> str:
            action = inputs.get("action")
            user_id = inputs.get("user_id")
            term = inputs.get("term", None)
            result = self.fetch_student_data(action=action, user_id=user_id, term=term)
            return str(result)

        # Create a list of tools - currently just the data fetching tool
        return [
            Tool(
                name="FetchStudentData",
                func=fetch_data,
                description="Obtiene datos específicos de estudiantes basados en la acción, ID de usuario y término.",
            )
        ]

    def classify_intent(self, message: str):
        """
        Classify the intent of a user message using the pre-trained ML models.
        
        Args:
            message (str): The user's message
            
        Returns:
            tuple: (predicted_intent, confidence_score)
        """
        try:
            # Clean and preprocess the input text
            clean_query = clean_text(message)
            
            # Transform the text to feature vector using the vectorizer
            query_vectorized = self.vectorizer.transform([clean_query])
            
            # Predict the intent class (numeric)
            predicted_label = self.classifier.predict(query_vectorized)[0]
            
            # Convert numeric label back to text label
            predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]
            
            # Get prediction probability/confidence
            probabilities = self.classifier.predict_proba(query_vectorized)[0]
            confidence = max(probabilities)

            logger.info(f"Predicted intent: {predicted_intent}, confidence: {confidence}")
            return predicted_intent, confidence
        except Exception as e:
            # Log and re-raise intent classification failures
            logger.error(f"Error classifying intent: {e}")
            raise RuntimeError("Failed to classify intent.")

    def process_with_agent(self, inputs: dict):
        """
        Process a query by calling the LLM API directly using the same prompt format.
        """
        try:
            # Extract the user's input from the inputs dictionary
            user_input = inputs.get("input", "")
            logger.info(f"Processing input: {user_input}")
            
            # First classify intent (we might use this to customize the prompt)
            intent, confidence = self.classify_intent(user_input)
            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")
            
            # Create enhanced user input while preserving exact prompt format
            enhanced_input = f"{user_input}\n\nIntención detectada: {intent} (confianza: {confidence:.2f})"
            
            raw_response = self._call_llm_api(enhanced_input)
            
            try:
                if isinstance(raw_response, dict):
                    return raw_response
                    
                json_output = json.loads(raw_response)
                logger.info(f"Parsed JSON output: {json_output}")
                return json_output
            except json.JSONDecodeError:
                logger.warning("LLM output is not valid JSON, wrapping raw text.")
                return {
                    "action": "respond",
                    "response": str(raw_response)
                }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": "Lo siento, no pude procesar tu consulta en este momento."}

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": "Lo siento, no pude procesar tu consulta en este momento."}