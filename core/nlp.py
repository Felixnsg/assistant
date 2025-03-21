##nlp
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from typing import Optional, Dict, Any, Union
import json


class LlpCall:
    def __init__(self, API_KEY= config.GEMINI_KEY):

        self.API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

        self.headers = { "content-type" : "application/json"}



    def send_request (self, data: str) -> Union[int, str]:
        """send the API request to the Gemini API endpoint and receive.

        Args:
            data (json): the formatted json request containing the prompt.

        Returns:
            str: The formatted response from Gemini, to only contain the text if successful request.
            int: int: The HTTP status code if the request fails.
        """

        try:
            response = requests.post(self.API_URL, json=data, headers=self.headers)
            if response.status_code == 200:
                try:
                    json_data = response.json()

                    if ('candidates' in json_data and 
                        json_data['candidates'] and 
                        'content' in json_data['candidates'][0] and
                        'parts' in json_data['candidates'][0]['content'] and
                        json_data['candidates'][0]['content']['parts'] and
                        'text' in json_data['candidates'][0]['content']['parts'][0]):

                        return json_data['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print(f"Different Structure Detected, please Consult the documentation")
                        return response.status_code
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    return response.status_code
        
            
            else:
                return response.status_code
            
        except (requests.ConnectionError, ConnectionError, requests.RequestException) as e:
            print(f"Big Error with the Call: {e}")
            return 500



