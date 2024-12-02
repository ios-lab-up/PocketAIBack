from app.models.chat import ChatResponse
from app.agents.base import StudentAgent
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def process_user_message(student_agent: StudentAgent, message: str) -> ChatResponse:
    """
    Process a user message by classifying intent and deciding on further action.

    Args:
        student_agent (StudentAgent): The initialized StudentAgent instance.
        message (str): The user message to process.

    Returns:
        ChatResponse: The response to send back to the user.
    """
    try:
        # Step 1: Classify the intent
        intent, confidence = student_agent.classify_intent(message)
        logger.info(f"Intent detected: {intent} with confidence: {confidence}")

        # Step 2: Decide next steps based on confidence
        if confidence > 0.8:
            try:
                # Use the LangChain agent to process the query
                agent_response = student_agent.process_with_agent(message)
                return ChatResponse(response=agent_response)
            except RuntimeError as agent_error:
                logger.error(f"Error processing with agent: {agent_error}")
                raise HTTPException(status_code=500, detail="Error processing query with LangChain agent.")
        else:
            # Fallback response for low-confidence classifications
            fallback_response = (
                "Lo siento, no estoy seguro de cómo ayudarte con esa consulta. "
                "¿Podrías darme más detalles?"
            )
            logger.info("Fallback response triggered due to low confidence.")
            return ChatResponse(response=fallback_response)

    except RuntimeError as e:
        # Handle errors during intent classification
        logger.error(f"Error classifying intent: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")
