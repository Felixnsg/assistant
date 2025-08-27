# File: utilities.py 
"""
Provides utility services for the assistant, such as fetching weather,
telling time, controlling mood lighting (YouTube via Selenium), and
managing the Felix video tracking client. Services are triggered based on
specific function triggers parsed from the AI's response.
"""

import asyncio
import os
import datetime
import requests
import sys
import json
import re
import time
import logging
from typing import Optional, Dict, Any, Tuple, Union

# Selenium imports with error handling
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.common.exceptions import WebDriverException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    logging.warning("Selenium library not found. Mood setter (YouTube) functionality will be disabled.")
    SELENIUM_AVAILABLE = False
    WebDriverException = Exception
    NoSuchElementException = Exception
    By = None

# Path setup and imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
    from core import nlp
except ImportError as e:
    logging.error(f"Error importing core/interface modules in utilities.py: {e}", exc_info=True)
    sys.exit(1)

# Import the video client and cache
try:
    from IseeYou.IseeYouClass import FlowControlledClient
    from core.cache import VisualContextCache
except ImportError as e:
    logging.error(f"Error importing video client or cache: {e}", exc_info=True)
    FlowControlledClient = None
    VisualContextCache = None

# Constants for triggers
TRIGGER_PREFIX = "FUNCTION_TRIGGER:"
SERVICE_GET_WEATHER = "GET_WEATHER"
SERVICE_TELL_TIME = "TELL_TIME"
SERVICE_SET_MOOD = "SET_MOOD"
SERVICE_START_VIDEO = "START_VIDEO"
SERVICE_STOP_VIDEO = "STOP_VIDEO"
SERVICE_CHECK_VISUAL_CONTEXT = "CHECK_VISUAL_CONTEXT"

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')

logger = logging.getLogger(__name__)

class Utilities:
    """
    Handles execution of various utility services triggered by the ChatManager.
    """
    def __init__(self,
                 config_instance: Any,
                 nlp_instance: Optional[nlp.LlpCall] = None,
                 chat_instance: Optional[Any] = None):
        """
        Initializes the Utilities class.

        Args:
            config_instance: The loaded configuration module or object.
            nlp_instance (Optional[nlp.LlpCall]): Instance for LLM calls.
            chat_instance (Optional[Any]): Instance of ChatManager.
        """
        logger.info("Initializing Utilities...")
        self.config = config_instance
        self.nlp = nlp_instance
        self.chat = chat_instance
        
        # Video service components, SIDE NOTE FUTURE ME, I didnt create it upfront because we dont need to start the system.
        self.video_client: Optional[FlowControlledClient] = None # type: ignore
        self.visual_cache: Optional[VisualContextCache] = None # type: ignore
        self.video_running = False
        
        # Selenium driver for mood setter/ I will remove this later and find a music api instead.
        self._selenium_driver: Optional[webdriver.Chrome] = None
        
        logger.info("Utilities initialized.")

    def tell_time(self) -> str:
        """
        Gets the current day, date, and time.

        Returns:
            str: A formatted string containing the current day, date, and time.
        """
        now = datetime.datetime.now()
        formatted_time = now.strftime("%A, %d %B %Y at %H:%M %p")
        logger.info(f"Generated time: {formatted_time}")
        return f"It is currently {formatted_time}."

    def get_weather(self, location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get weather information for a location using WeatherAPI.com.

        Args:
            location (Optional[str]): City name or location query.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing weather information.
        """
        target_location = location if location else self.config.DEFAULT_WEATHER_LOCATION
        api_key = self.config.WEATHER_API_KEY
        if not api_key:
            logger.error("WeatherAPI key not configured in config.py.")
            return None

        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={target_location}&days=1&aqi=no&alerts=no"
        logger.info(f"Fetching weather for '{target_location}'...")

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            jsonData = response.json()

            # Extract relevant data
            current = jsonData.get("current", {})
            forecast_day = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
            location_info = jsonData.get("location", {})
            astro_info = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("astro", {})

            weather_info = {
                "location": f"{location_info.get('name', 'N/A')}, {location_info.get('region', 'N/A')}, {location_info.get('country', 'N/A')}",
                "time": location_info.get('localtime', 'N/A'),
                "temp_c": current.get('temp_c'),
                "temp_f": current.get('temp_f'),
                "feelslike_c": current.get('feelslike_c'),
                "feelslike_f": current.get('feelslike_f'),
                "condition": current.get('condition', {}).get('text', 'N/A'),
                "humidity": current.get('humidity'),
                "wind_kph": current.get('wind_kph'),
                "wind_mph": current.get('wind_mph'),
                "wind_dir": current.get('wind_dir'),
                "precip_mm": current.get('precip_mm'),
                "uv_index": current.get('uv'),
                "forecast_max_c": forecast_day.get('maxtemp_c'),
                "forecast_min_c": forecast_day.get('mintemp_c'),
                "forecast_max_f": forecast_day.get('maxtemp_f'),
                "forecast_min_f": forecast_day.get('mintemp_f'),
                "sunrise": astro_info.get('sunrise'),
                "sunset": astro_info.get('sunset')
            }
            logger.info(f"Weather fetched successfully for '{target_location}'.")
            return weather_info

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching weather: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in get_weather: {e}", exc_info=True)
            return None

    def _start_youtube_mood(self) -> bool: ##this will leave very soon just give me time.
        """
        Opens a predefined YouTube video using Selenium.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Cannot start mood setter: Selenium library is not available.")
            return False
        
        if self._selenium_driver is not None:
            logger.warning("Mood setter already running.")
            self._stop_youtube_mood()

        url = self.config.YOUTUBE_MOOD_URL
        logger.info(f"Starting mood setter: {url}")

        try:
            options = webdriver.ChromeOptions()
            # options.add_argument("--headless")  # Uncomment for headless mode
            self._selenium_driver = webdriver.Chrome(options=options)
            self._selenium_driver.get(url)
            time.sleep(5)

            # Try to click play button
            play_button_selector = "button.ytp-play-button[aria-label*='Play']"
            try:
                play_button = self._selenium_driver.find_element(By.CSS_SELECTOR, play_button_selector)
                play_button.click()
                logger.info("Clicked YouTube play button.")
            except NoSuchElementException:
                logger.warning("Could not find play button - video might be auto-playing.")
            
            return True

        except WebDriverException as e:
            logger.error(f"Selenium WebDriver error: {e}")
            self._selenium_driver = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting mood setter: {e}", exc_info=True)
            if self._selenium_driver:
                self._selenium_driver.quit()
            self._selenium_driver = None
            return False

    def _stop_youtube_mood(self) -> bool: #yeah sorry this will leave soon, past me was thinking too far.
        """Quits the Selenium browser instance if running."""
        if self._selenium_driver:
            logger.info("Stopping mood setter.")
            try:
                self._selenium_driver.quit()
                self._selenium_driver = None
                return True
            except Exception as e:
                logger.error(f"Error stopping mood setter: {e}")
                self._selenium_driver = None
                return False
        return True

    async def start_video_service(self) -> bool:
        """
        Starts the video tracking service with cache integration.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.video_running:
            logger.warning("Video service is already running.")
            return True
        
        if not FlowControlledClient or not VisualContextCache:
            logger.error("Video client or cache classes not available.")
            return False
        
        try:
            # Create cache instance
            self.visual_cache = VisualContextCache()
            
            # Create video client with cache callback
            self.video_client = FlowControlledClient(
                server_uri="ws://localhost:8080",
                target_fps=30,
                cache_callback=self.visual_cache.update_from_client
            )
            
            # Start the client in background
            self.video_task = asyncio.create_task(self.video_client.start())
            self.video_running = True
            
            logger.info("Video service started successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video service: {e}", exc_info=True)
            self.video_running = False
            self.video_client = None
            self.visual_cache = None
            return False

    async def stop_video_service(self) -> bool:
        """
        Stops the video tracking service gracefully.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.video_running:
            logger.info("Video service not running.")
            return True
        
        logger.info("Stopping video service...")
        self.video_running = False
        
        try:
            # Stop the video client
            if self.video_client:
                await self.video_client.stop()
                self.video_client = None
            
            # Clear cache
            if self.visual_cache:
                await self.visual_cache.clear()
                self.visual_cache = None
            
            logger.info("Video service stopped successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping video service: {e}", exc_info=True)
            return False

    async def get_visual_context(self) -> Dict[str, Any]:
        """
        Gets the current visual context from the cache.
        
        Returns:
            Dict with 'context_string' key containing the formatted context.
        """
        if not self.visual_cache:
            return {"context_string": "[Visual context: Vision system not active]"}
        
        try:
            context_string = await self.visual_cache.get_visual_context_string()
            return {"context_string": context_string}
        except Exception as e:
            logger.error(f"Error getting visual context: {e}")
            return {"context_string": "[Visual context: Error retrieving vision data]"}

    async def dispatch_service(self, ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Parses the AI response for function triggers and executes the corresponding service.

        Args:
            ai_response (str): The response text from the AI.

        Returns:
            Optional[Dict[str, Any]]: Service result or None.
        """
        if not isinstance(ai_response, str):
            return None

        match = re.search(rf"{TRIGGER_PREFIX}(\w+)(?::(.*))?", ai_response)
        if not match:
            return None

        service_name = match.group(1).upper()
        parameter = match.group(2) if match.group(2) else None
        logger.info(f"Detected trigger: Service='{service_name}', Parameter='{parameter}'")

        result_payload: Dict[str, Any] = {"service": service_name, "result": None}

        try:
            if service_name == SERVICE_GET_WEATHER:
                weather_data = self.get_weather(location=parameter)
                result_payload["result"] = weather_data
                
            elif service_name == SERVICE_TELL_TIME:
                time_string = self.tell_time()
                result_payload["result"] = time_string
                
            elif service_name == SERVICE_SET_MOOD:
                success = self._start_youtube_mood()
                result_payload["result"] = success
                
            elif service_name == SERVICE_START_VIDEO:
                success = await self.start_video_service()
                result_payload["result"] = success
                
            elif service_name == SERVICE_STOP_VIDEO:
                success = await self.stop_video_service()
                result_payload["result"] = success
                
            elif service_name == SERVICE_CHECK_VISUAL_CONTEXT:
                context_data = await self.get_visual_context()
                result_payload["result"] = context_data
                
            else:
                logger.warning(f"Unknown service trigger: {service_name}")
                return None

            logger.info(f"Service '{service_name}' executed successfully.")
            return result_payload

        except Exception as e:
            logger.error(f"Error executing service '{service_name}': {e}", exc_info=True)
            return None

    def cleanup(self):
        """Cleans up resources used by Utilities."""
        logger.info("Cleaning up Utilities resources...")
        self._stop_youtube_mood()
        logger.info("Utilities cleanup finished.")

    # These methods maintain compatibility with existing code
    async def start(self):
        """Compatibility method for starting video service."""
        return await self.start_video_service()

    async def stop(self):
        """Compatibility method for stopping video service."""
        return await self.stop_video_service()

    async def exit(self):
        """Compatibility method for stopping video service."""
        return await self.stop_video_service()