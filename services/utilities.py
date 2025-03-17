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


def tell_time():
    x = datetime.datetime.now()
    a = x.strftime("%A") 
    return a, x  


def monitor_sypher(ai_response):
    keywords = [
        "I will check the weather for you",
        "to get the weather I would need",
        "Wait, I will pull that up for you.",
        "Okay, switching back now."
    ]

    # Directly check if any phrase is in ai_response
    for phrase in keywords:
        if phrase in ai_response:
            return True
    return False





def choose_service (ai_response):
    if monitor_sypher(ai_response):
        print("choosing format................")
        if ai_response == "He chose the weather":
            get_weather()
        else:
            print("setting the mood..............")
            driver = mood_setter()
            switch_mode(ai_response)
            for i in range(config.NUMBER_STORIES):
                switch_mode_2(ai_response)

            while streamaudio.has_items_in_queue():
                time.sleep(1)
            time.sleep(5)

            print("Leave Mood...........")

            driver.quit()









def switch_mode(ai_response):
    print(f"{config.MODEL_NAME}: switching mode..................")
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    convos.append({"role" : "user" , "parts" : [{"text" : config.switch_mode_prompt}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
    ai_response = nlp.send_request(data)
    streamaudio.say(ai_response)
    memory.save_convos("user", config.switch_mode_prompt)
    memory.save_convos("model", ai_response)
    

    print(f"{config.MODEL_NAME} {ai_response}")





def switch_mode_2(ai_response):
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    convos.append({"role" : "user" , "parts" : [{"text" : config.switch_mode_prompt_2}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
    ai_response = nlp.send_request(data)
    streamaudio.say(ai_response)
    memory.save_convos("user", config.switch_mode_prompt_2)
    memory.save_convos("model", ai_response)

    print(f"{config.MODEL_NAME} {ai_response}")



def mood_setter():
    url = "https://www.youtube.com/watch?v=ztVV54sPOns&t=461s"
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)

    play_button = driver.find_element(By.CLASS_NAME, "ytp-play-button")
    play_button.click()

    return driver












def weather_service ():
    print(f"{config.MODEL_NAME}: Let me pull up that for you, gimme a minute....")
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    weather_info = get_weather()

    convos.append({"role" : "user" , "parts" : [{"text" : config.weather_prompt + str(weather_info)}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
        
    ai_response = nlp.send_request(data)  

    print(ai_response)
        











                
import requests
import sys

def get_weather(location="seattle", api_key="66ff43b4ee214eed86763359251603"):
    """
    Get weather information for a location using WeatherAPI.com
    
    Parameters:
    - location (str): City name or location query
    - api_key (str): Your WeatherAPI.com API key
    
    Returns:
    - dict: Weather information
    """
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
    weather_info = {
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
    
    return weather_info

