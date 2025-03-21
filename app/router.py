import logging
from fastapi import APIRouter, HTTPException, Response, Query
from fastapi.responses import JSONResponse
from app.models.chat import ChatRequest, ChatResponse, PredictResponse
from app.agents.base import StudentAgent
from app.agents.intent_agent import IntentBasedAgent
import time
import joblib
import requests



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    vectorizer = joblib.load("../data/models/vectorizer.pkl")
    classifier = joblib.load("../data/models/intent_classifier.pkl")
    label_encoder = joblib.load("../data/models/label_encoder.pkl")
    logger.info("Model and artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model artifacts: {e}")
    raise e

student_agent = StudentAgent()
router = APIRouter()

@router.get("/health")
def healthcheck(response: Response):
    """
    Check the health of the API, provide server time, a random server ID, and the client's IP address.
    """
    server_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    server_id = 1
    version = "1.0.1"


    # Attempt to fetch the public IP address
    try:
        public_ip = requests.get("https://api.ipify.org?format=json", timeout=10).json()
    except requests.RequestException:
        public_ip = 'Unavailable'

    logging.info(f"Health check accessed at {server_time} with server ID {server_id} from IP {public_ip}")


    return {
        "status": "ok",
        "server_time": server_time,
        "server_id": server_id,
        **public_ip,
        "version": version
    }
@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Query(..., description="The user ID of the student"),
    term: str = Query(None, description="Optional term parameter for context")
):
    """
    Process a user query and respond using the IntentBasedAgent class,
    with additional context such as user_id and term.
    """
    logger.info(f"Received message: {request.message}, User ID: {user_id}, Term: {term}")

    try:
        # Use the IntentBasedAgent class directly
        response = IntentBasedAgent.handle_query(
            user_query=request.message,
            user_id=user_id,
            term=term
        )
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return ChatResponse(response="Lo siento, no pude procesar tu consulta en este momento.")

# @router.post("/predict", response_model=PredictResponse)
# async def predict_intent(request: ChatRequest):
#     """
#     Predict the intent of a given query.
#     """
#     try:
#         # Preprocess the user's query
#         clean_query = clean_text(request.message)

#         # Vectorize the query
#         query_vectorized = vectorizer.transform([clean_query])

#         # Predict the intent
#         predicted_label = classifier.predict(query_vectorized)[0]
#         predicted_intent = label_encoder.inverse_transform([predicted_label])[0]

#         # Get confidence score
#         probabilities = classifier.predict_proba(query_vectorized)[0]
#         confidence = max(probabilities)

#         logger.info(f"Predicted intent: {predicted_intent} with confidence: {confidence}")
#         return PredictResponse(intent=predicted_intent, confidence=confidence)
#     except Exception as e:
#         logger.error(f"Error predicting intent: {e}")
#         raise HTTPException(status_code=500, detail="Failed to predict intent.")
