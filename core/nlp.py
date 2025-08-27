## nlp
import requests
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
except ImportError as e:
    print(f"Error importing config module: {e}. Ensure config.py exists and is accessible.")
    sys.exit(1)

from typing import Optional, Dict, Any, Union, List # Added List


BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent" 

class LlpCall:
    """
    Handles interactions with the configured Large Language Model (LLM) API.

    Specifically designed for Google's Generative AI API (Gemini).

    Attributes:
        api_key (str): The API key for accessing the LLM service.
        api_url (str): The full URL endpoint for the LLM API.
        headers (dict): HTTP headers required for the API request.
    """
    def __init__(self, api_key: str = config.GEMINI_KEY):
        """
        Initializes the LlpCall instance.

        Args:
            api_key (str): The API key for the Generative AI service.
                           Defaults to config.GEMINI_KEY.

        Raises:
            ValueError: If the API key is not provided or invalid.
        """
        if not api_key:
            raise ValueError("LLM API key is missing or invalid. Please check config.py or .env file.")

        self.api_key = api_key
        self.api_url = f"{BASE_API_URL}?key={self.api_key}"
        self.headers = {"content-type": "application/json"}
        print(f"LLM Caller initialized for model endpoint: {BASE_API_URL}")

    def send_request(self, data: Dict[str, Any]) -> Union[str, int]:
        """
        Sends the prepared data as a request to the LLM API endpoint.

        Args:
            data (Dict[str, Any]): The request payload, typically containing
                                   conversation history ('contents') and
                                   generation configuration. Expected to be in
                                   the format required by the Google Generative AI API.

        Returns:
            Union[str, int]: The extracted text response from the LLM if successful (status code 200),
                             otherwise returns the HTTP status code (int) indicating the error.
                             Returns 500 for connection or unexpected request errors.
        """
        if not isinstance(data, dict) or 'contents' not in data:
            print("Error: Invalid data format for LLM request. 'contents' key missing.")
            return 400

        try:
            response = requests.post(self.api_url, json=data, headers=self.headers, timeout=60)

            if response.status_code == 200:
                try:
                    json_data = response.json()

                    # Check for potential blocking due to safety settings
                    if 'promptFeedback' in json_data and 'blockReason' in json_data['promptFeedback']:
                        reason = json_data['promptFeedback']['blockReason']
                        print(f"Warning: LLM request blocked due to safety settings. Reason: {reason}")
                        details = json_data['promptFeedback'].get('safetyRatings', [])
                        print(f"Details: {details}")
                        return f"Blocked: {reason}"

                    # Check candidates list and content parts
                    if ('candidates' in json_data and
                            isinstance(json_data['candidates'], list) and
                            len(json_data['candidates']) > 0 and
                            'content' in json_data['candidates'][0] and
                            'parts' in json_data['candidates'][0]['content'] and
                            isinstance(json_data['candidates'][0]['content']['parts'], list) and
                            len(json_data['candidates'][0]['content']['parts']) > 0 and
                            'text' in json_data['candidates'][0]['content']['parts'][0]):

                        return json_data['candidates'][0]['content']['parts'][0]['text'].strip()
                    elif 'candidates' in json_data and not json_data.get('candidates'):
                         print("LLM response received, but 'candidates' list is empty.")
                         return "" 
                    else:
                        print("LLM response structure unexpected. Dumping response:")
                        print(json.dumps(json_data, indent=2))
                        return 500 # Internal Server Error (unexpected response format)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON response from LLM: {e}")
                    print(f"Response text: {response.text[:500]}...") # Log partial raw response
                    return response.status_code # Return original status code
                except (KeyError, IndexError, TypeError) as e:
                     print(f"Error parsing LLM JSON response structure: {e}")
                     print(json.dumps(json_data, indent=2)) # Log the structure that caused the error
                     return 500 # Internal Server Error (parsing failed)

            else:
                print(f"LLM API request failed with status code: {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"API Error Details: {json.dumps(error_details, indent=2)}")
                except json.JSONDecodeError:
                    print(f"API Error Response (non-JSON): {response.text[:500]}...")
                return response.status_code

        except requests.exceptions.Timeout:
            print(f"Error: LLM API request timed out after {response.timeout} seconds.")
            return 408 # Request Timeout
        except requests.exceptions.ConnectionError as e:
            print(f"Error: Could not connect to LLM API at {self.api_url}. Details: {e}")
            return 503 # Service Unavailable
        except requests.exceptions.RequestException as e:
            # Catch other potential requests errors (like InvalidURL, TooManyRedirects)
            print(f"An unexpected error occurred during the LLM API request: {e}")
            return 500 # Internal Server Error
        except Exception as e:
             # Catch any other unexpected errors
             print(f"An unexpected error occurred in send_request: {e}")
             # import traceback # Optionally add traceback here for debugging
             # traceback.print_exc()
             return 500 # Internal Server Error