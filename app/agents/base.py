from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from app.services.student_data import StudentDataAPI
from app.utils.preprocess import clean_text
import joblib
import logging

logger = logging.getLogger(__name__)


class StudentAgent:

    def __init__(self, vectorizer_path="app/models/vectorizer.pkl",
                 classifier_path="app/models/intent_classifier.pkl",
                 label_encoder_path="app/models/label_encoder.pkl",
                 llm_model="gpt-3.5-turbo"):
        """
        Initialize the StudentAgent, including models and LangChain agent setup.

        Args:
            vectorizer_path (str): Path to the vectorizer pickle file.
            classifier_path (str): Path to the classifier pickle file.
            label_encoder_path (str): Path to the label encoder pickle file.
            llm_model (str): Language model to use (default: "gpt-3.5-turbo").
        """
        self.llm_model = llm_model

        # Load intent classification models
        self._load_models(vectorizer_path, classifier_path, label_encoder_path)

        # Initialize LangChain agent
        self.agent = self._initialize_agent()

    def _load_models(self, vectorizer_path, classifier_path, label_encoder_path):
        """
        Load the intent classification models.
        """
        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(classifier_path)
            self.label_encoder = joblib.load(label_encoder_path)
            logger.info("Intent classification models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise RuntimeError("Failed to load intent classification models.")

    def _initialize_agent(self):
        """
        Initialize the LangChain agent with tools.

        Returns:
            agent: The initialized LangChain agent.
        """
        try:
            # Initialize StudentDataAPI
            student_api = StudentDataAPI()

            # Define LangChain tools
            tools = student_api.define_tools()

            # Initialize the LLM
            llm = ChatOpenAI(temperature=0.7, model=self.llm_model)

            # Initialize the LangChain agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            logger.info("LangChain agent initialized successfully.")
            return agent
        except Exception as e:
            logger.error(f"Error initializing LangChain agent: {e}")
            raise RuntimeError("Failed to initialize LangChain agent.")

    def classify_intent(self, message: str):
        """
        Classify the intent of a message using the intent classifier.

        Args:
            message (str): The user query.

        Returns:
            tuple: (predicted_intent, confidence)
        """
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

    def process_with_agent(self, message: str):
        """
        Process a message with the LangChain agent.

        Args:
            message (str): The user query.

        Returns:
            str: The agent's response.
        """
        try:
            response = self.agent.run(message)
            logger.info(f"Agent response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing query with LangChain agent: {e}")
            # Graceful fallback
            return "Lo siento, no pude procesar tu consulta en este momento. Por favor, intenta nuevamente."
