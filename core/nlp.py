##nlp

import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config




API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={config.GEMINI_KEY}"

headers = { "content-type" : "application/json"}

def send_request (data):
    
    response = requests.post(API_URL, json=data, headers=headers)
    if response.status_code == 200:
        json_data = response.json()
        return json_data['candidates'][0]['content']['parts'][0]['text']
    else:
        return response.status_code
