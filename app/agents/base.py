from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, Tool
from langchain.prompts import PromptTemplate
from app.utils.preprocess import clean_text
from langchain_core.output_parsers import JsonOutputParser

import joblib
import requests
import logging
import json
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    action: str = Field(description="El nombre de la acción a realizar.")
    response: str = Field(description="La respuesta que se devolverá al usuario.")

class StudentAgent:
    """
    Agent for student queries that integrates intent classification,
    data retrieval, and response generation using LangChain.
    """

    def __init__(self, vectorizer_path="data/models/vectorizer.pkl",
                 classifier_path="data/models/intent_classifier.pkl",
                 label_encoder_path="data/models/label_encoder.pkl",
                 llm_model="llama3.2:3b",
                 base_url=None, timeout=10):
        """
        Initialize the StudentAgent, including models and LangChain agent setup.

        Args:
            vectorizer_path (str): Path to the vectorizer pickle file.
            classifier_path (str): Path to the classifier pickle file.
            label_encoder_path (str): Path to the label encoder pickle file.
            llm_model (str): Language model to use (default: "gpt-3.5-turbo").
            base_url (str): Base URL for the student data API.
            timeout (int): Timeout for the student data API requests.
        """
        self.llm_model = llm_model
        self.base_url = base_url
        self.timeout = timeout
        self.parser = JsonOutputParser(pydantic_object=AgentResponse)

        # Load intent classification models
        self._load_models(vectorizer_path, classifier_path, label_encoder_path)

        # Initialize LangChain agent
        self.agent = self._initialize_agent()

    def _load_models(self, vectorizer_path, classifier_path, label_encoder_path):
        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(classifier_path)
            self.label_encoder = joblib.load(label_encoder_path)
            logger.info("Intent classification models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError("Failed to load intent classification models.")

    def fetch_student_data(self, action: str, user_id: str, term: str = None) -> dict:
        try:
            params = {"action": action, "user_id": user_id}
            if term:
                params["term"] = term

            response = requests.get(
                f"{self.base_url}/student-data",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch student data: {e}")
            return {"error": f"Failed to fetch student data: {str(e)}"}
    def _initialize_agent(self):
        """
        Initialize the LangChain agent with tools.
        """
        try:
            tools = self._define_tools()
            logger.info(f"Tools successfully defined: {tools}")

            llm = ChatOpenAI(temperature=0.7, model=self.llm_model)
            logger.info(f"LLM initialized with model: {self.llm_model}")

            # Generar los nombres de las herramientas
            tool_names = ", ".join([tool.name for tool in tools])

            format_instructions = self.parser.get_format_instructions()

            # Crear el prompt con las variables requeridas
            prompt = PromptTemplate(
                input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
                template=(
                    "Eres un asistente útil que siempre responde en JSON. Responde a la siguiente entrada en este formato estricto:\n"
                    f"{format_instructions}\n\n"
                    "Herramientas disponibles:\n"
                    "{tools}\n\n"
                    "Nombres de herramientas:\n"
                    "{tool_names}\n\n"
                    "Acciones realizadas hasta ahora:\n"
                    "{agent_scratchpad}\n\n"
                    "Entrada del usuario:\n"
                    "{input}\n"
                ),
            )

            # Inicializar el agente con el prompt y las herramientas
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt,
            )
            logger.info("LangChain agent initialized successfully.")
            return agent
        except Exception as e:
            logger.error(f"Error initializing LangChain agent: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize LangChain agent.")


    def _define_tools(self) -> list:
        def fetch_data(inputs: dict) -> str:
            action = inputs.get("action")
            user_id = inputs.get("user_id")
            term = inputs.get("term", None)
            result = self.fetch_student_data(action=action, user_id=user_id, term=term)
            return str(result)

        return [
            Tool(
                name="FetchStudentData",
                func=fetch_data,
                description="Obtiene datos específicos de estudiantes basados en la acción, ID de usuario y término.",
            )
        ]

    def classify_intent(self, message: str):
        try:
            clean_query = clean_text(message)
            query_vectorized = self.vectorizer.transform([clean_query])
            predicted_label = self.classifier.predict(query_vectorized)[0]
            predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]
            probabilities = self.classifier.predict_proba(query_vectorized)[0]
            confidence = max(probabilities)

            logger.info(f"Predicted intent: {predicted_intent}, confidence: {confidence}")
            return predicted_intent, confidence
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            raise RuntimeError("Failed to classify intent.")

    def process_with_agent(self, inputs: dict):
        try:
            formatted_input = inputs.get("input", "")
            logger.info(f"Formatted input for LangChain agent: {formatted_input}")

            raw_response = self.agent.invoke({"input": formatted_input})
            logger.info(f"Raw agent response: {raw_response}")

            try:
                json_output = json.loads(raw_response)
                logger.info(f"Parsed JSON output: {json_output}")
                return json_output
            except json.JSONDecodeError:
                logger.warning("LLM output is not a valid JSON, returning raw text.")
                return raw_response

        except Exception as e:
            logger.error(f"Error processing query with LangChain agent: {e}")
            return {"error": "Lo siento, no pude procesar tu consulta en este momento."}
