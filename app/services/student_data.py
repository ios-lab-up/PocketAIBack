from langchain.agents import Tool
from app.core.settings import settings  # Import the settings
import requests


class StudentDataAPI:
    """
    A class to encapsulate interactions with the student data API.
    Provides methods to fetch student data and define LangChain tools.
    """

    def __init__(self):
        """
        Initialize the API client using environment variables.
        """
        self.base_url = settings.API_BASE_URL
        self.timeout = settings.TIMEOUT

    def fetch_student_data(self, action: str, user_id: str, term: str = None) -> dict:
        """
        Fetch student data for the given action.

        Args:
            action (str): The specific action (e.g., "grades", "attendance", "schedule").
            user_id (str): The user identifier (e.g., "0250009").
            term (str, optional): Optional term for term-specific actions.

        Returns:
            dict: The JSON response from the API or an error message.
        """
        try:
            response = requests.get(
                f"{self.base_url}/student-data",
                params={"action": action, "user_id": user_id, "term": term},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch student data: {str(e)}"}

    def fetch_general_average(self, user_id: str) -> dict:
        """
        Fetch the general average grades for the student.

        Args:
            user_id (str): The user identifier.

        Returns:
            dict: The JSON response from the API or an error message.
        """
        try:
            response = requests.get(
                f"{self.base_url}/general-average",
                params={"user_id": user_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch general average: {str(e)}"}

    def define_tools(self) -> list:
        """
        Define LangChain tools for interacting with the student data API.

        Returns:
            list: A list of LangChain tools.
        """
        return [
            Tool(
                name="FetchStudentData",
                func=lambda action, user_id, term=None: self.fetch_student_data(
                    action, user_id, term
                ),
                description="Fetch specific student data based on the action (e.g., grades, attendance, schedule).",
            ),
            Tool(
                name="FetchGeneralAverage",
                func=lambda user_id: self.fetch_general_average(user_id),
                description="Fetch the general average grades for a student.",
            ),
        ]
