import datetime
import requests
import sys
import json
from core import memory
import config
from core import nlp
from interfaces import streamaudio
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import spotipy
import json
import webbrowser
from interfaces import chat



class Utilities:
    def __init__(self, Chat_instance, Config_instance, nlp_instance):

        self.nlp = nlp_instance
        self.chat = Chat_instance
        self.config = Config_instance
        self.location = "Seattle"
        self.weather_info = ""
        self.convos = []

        

    def tell_time(self):
        """Uses the datetime module to return the system's current time and date.

        Returns:
        tuple[str, datetime.datetime]: A tuple containing the weekday name 
        and the current date-time object.
        """

        x = datetime.datetime.now()
        a = x.strftime("%A") 

        return a, x  


    def monitor_sypher(self, ai_response):
        """This function is a boolean function that returns True
        if a certain condition is met.

        Args:
            ai_response (str): This parameter, when called will contain the ai response text.

        Returns:
            boolean: True or False depending on where or when called.
        """
       
        
        keywords = [
            "I will check the weather for you",
            "to get the weather I would need",
            "Wait, I will pull that up for you.",
            "Okay, switching back now."
        ]

        for phrase in keywords:
            if phrase in ai_response:
                return True
        return False


    def choose_service (self, ai_response):
        """This function toggle a different function the assistant each time, 
        A different function is called according to the condition.

        Args:
            ai_response (str): This parameter when called, is designed to contain the ai text.
        """
        
        if self.monitor_sypher(ai_response):
            print("choosing format................")

            if ai_response == "He chose the weather":
                self.get_weather()
        
            else:
                print("setting the mood..............")
                driver = self.mood_setter()
                self.switch_mode(ai_response)

                for i in range(self.config.NUMBER_STORIES):
                    self.switch_mode_2(ai_response)
                try:
                    while streamaudio.has_items_in_queue():
                        time.sleep(1)
                        time.sleep(5)
                except Exception as e:
                    print(f"Issue with Streamaudio: {e} ")
                print("Leave Mood...........")

                driver.quit()


    def switch_mode(self, ai_response):
        """Handles mode switching for the AI assistant.

        This function:
        - Logs the mode switch action.
        - Retrieves and updates past conversations.
        - Sends a request to the NLP model with the updated conversation.
        - Converts the AI response to speech.
        - Saves the conversation history.

        Args:
            ai_response (str): The AI-generated response to be processed.

        Returns:
            None: The function prints the response, plays audio, and updates memory.
        """


        print(f"{self.config.MODEL_NAME}: switching mode..................")
        data = chat.data_prep(self.config.switch_mode_prompt, self.convos)
        response = self.nlp.send_request(data)
        print(f"ma_boi: {response}")
        streamaudio.say(response)





    def switch_mode_2(self, ai_response):
        """Handles an alternate mode switch for the AI assistant.

        This function:
        - Retrieves and updates past conversations.
        - Sends a request to the NLP model with the updated context.
        - Converts the AI response to speech.
        - Saves the conversation history.

        Args:
            ai_response (str): The AI-generated response to be processed.

        Returns:
            None: Prints the response, plays audio, and updates memory.
        """


        data = chat.data_prep(self.config.switch_mode_prompt_2, self.convos)
        response = self.nlp.send_request(data)
        print(f"ma_boi: {response}")
        streamaudio.say(response)



    def mood_setter(self):
        """Opens a YouTube video to set the mood.

        This function:
        - Launches a Chrome browser instance.
        - Navigates to a predefined YouTube video.
        - Clicks the play button to start playback.

        Returns:
            WebDriver: The Selenium WebDriver instance controlling the browser.
        """


        url = "https://www.youtube.com/watch?v=ztVV54sPOns&t=461s"
        driver = webdriver.Chrome()
        driver.get(url)
        time.sleep(3)

        play_button = driver.find_element(By.CLASS_NAME, "ytp-play-button")
        play_button.click()

        return driver

    def weather_service(self):
        """Retrieves weather information and interacts with the AI assistant.

        This function:
        - Fetches the latest weather data.
        - Updates the conversation history with the weather information.
        - Sends a request to the NLP model with the weather context.
        - Prints the AI-generated weather response.

        Returns:
            None: The function prints the response but does not return data.
        """


        print(f"{self.config.MODEL_NAME}: Let me pull up that for you, gimme a minute....")
        data = chat.data_prep(self.config.weather_prompt + self.weather_info, self.convos)
        response = self.nlp.send_request(data)
        print(f"{self.config.MODEL_NAME}: {response}")
        streamaudio.say(response)

        

        
    def get_weather(self, api_key="66ff43b4ee214eed86763359251603"):
        """Get weather information for a location using WeatherAPI.com
        
        Parameters:
        - location (str): City name or location query
        - api_key (str): Your WeatherAPI.com API key
        
        Returns:
        - dict: Weather information
        """
        location = self.location
        # Build the API URL
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1&aqi=no&alerts=no"
        
        # Make the request
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code != 200:
            print('Unexpected Status code:', response.status_code)
            if response.text:
                print("Error details:", response.text)
            sys.exit()
        
        # Parse the results as JSON
        jsonData = response.json()
        
        # Get current weather data
        current = jsonData.get("current", {})
        
        # Get forecast data for today
        forecast_day = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
        
        # Get location information
        location_info = jsonData.get("location", {})
        
        # Get astronomical data
        astro_info = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("astro", {})
        
        # Create a dictionary with weather information
        self.weather_info = {
            "Location": f"{location_info.get('name', 'N/A')}, {location_info.get('country', 'N/A')}",
            "Date": location_info.get('localtime', 'N/A'),
            "Current Temperature": f"{current.get('temp_c', 'N/A')}°C ({current.get('temp_f', 'N/A')}°F)",
            "Feels Like": f"{current.get('feelslike_c', 'N/A')}°C ({current.get('feelslike_f', 'N/A')}°F)",
            "Max Temperature": f"{forecast_day.get('maxtemp_c', 'N/A')}°C ({forecast_day.get('maxtemp_f', 'N/A')}°F)",
            "Min Temperature": f"{forecast_day.get('mintemp_c', 'N/A')}°C ({forecast_day.get('mintemp_f', 'N/A')}°F)",
            "Weather Description": current.get('condition', {}).get('text', 'N/A'),
            "Humidity": f"{current.get('humidity', 'N/A')}%",
            "Wind": f"{current.get('wind_kph', 'N/A')} km/h ({current.get('wind_mph', 'N/A')} mph)",
            "Wind Direction": current.get('wind_dir', 'N/A'),
            "Precipitation": f"{current.get('precip_mm', 'N/A')} mm ({current.get('precip_in', 'N/A')} in)",
            "UV Index": current.get('uv', 'N/A'),
            "Sunrise": astro_info.get('sunrise', 'N/A'),
            "Sunset": astro_info.get('sunset', 'N/A')
        }
        
        return self.weather_info