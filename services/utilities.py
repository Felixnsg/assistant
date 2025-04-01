# File: utilities.py
# --- REFACTOR: Added file description ---
"""
Provides utility services for the assistant, such as fetching weather,
telling time, controlling mood lighting (YouTube via Selenium), and
managing the Felix video tracking client. Services are triggered based on
specific function triggers parsed from the AI's response.
"""
import os
import datetime
import requests
import sys
import json
import re # --- REFACTOR: Added re for trigger parsing ---
import time
import logging # --- REFACTOR: Added logging ---
from typing import Optional, Dict, Any, Tuple, Union # --- REFACTOR: Added typing ---

# --- REFACTOR: Selenium imports with error handling ---
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service as ChromeService # Optional: For specific driver path
    from selenium.common.exceptions import WebDriverException, NoSuchElementException
    # from webdriver_manager.chrome import ChromeDriverManager # Optional: For auto driver download
    SELENIUM_AVAILABLE = True
except ImportError:
    logging.warning("Selenium library not found. Mood setter (YouTube) functionality will be disabled.")
    SELENIUM_AVAILABLE = False
    # Define dummy classes/variables if needed to prevent NameErrors later
    WebDriverException = Exception
    NoSuchElementException = Exception
    By = None # type: ignore

# --- REFACTOR: Simplified path appending and imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
    from core import nlp # Assumed LlpCall class is needed if follow-up calls remained (they are removed for now)
    # from interfaces import streamaudio # Assumed needed for .say() if used directly (removed for now)
    # --- REFACTOR: Import Felix client for control ---
    from IseeYou.IseeYou import FelixTrackingClient # Import the class definition
except ImportError as e:
    logging.error(f"Error importing core/interface modules in utilities.py: {e}", exc_info=True)
    # Decide if this is fatal - likely yes if ChatManager depends on it
    sys.exit(1)

# --- REFACTOR: Constants for triggers ---
TRIGGER_PREFIX = "FUNCTION_TRIGGER:"
SERVICE_GET_WEATHER = "GET_WEATHER"
SERVICE_TELL_TIME = "TELL_TIME"
SERVICE_SET_MOOD = "SET_MOOD"
SERVICE_START_VIDEO = "START_VIDEO"
SERVICE_STOP_VIDEO = "STOP_VIDEO"
# Add other services like SPOTIFY here if implemented

# --- REFACTOR: Configure logging ---
# Logging setup might be better in main.py, but adding basic config here for standalone usability
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')


class Utilities:
    """
    Handles execution of various utility services triggered by the ChatManager.
    """
    def __init__(self,
                 config_instance: Any,
                 nlp_instance: Optional[nlp.LlpCall] = None, # Optional, may not be needed directly
                 chat_instance: Optional[Any] = None): # Optional, avoid tight coupling if possible
        """
        Initializes the Utilities class.

        Args:
            config_instance: The loaded configuration module or object.
            nlp_instance (Optional[nlp.LlpCall]): Instance for LLM calls (potentially removed).
            chat_instance (Optional[Any]): Instance of ChatManager (potentially removed).
        """
        logging.info("Initializing Utilities...")
        self.config = config_instance
        self.nlp = nlp_instance # Store if needed, but aim to remove direct use
        self.chat = chat_instance # Store if needed, but aim to remove direct use

        # --- REFACTOR: Removed self.location and self.weather_info ---
        # Location should be passed per request, weather info returned by function

        self._selenium_driver: Optional[webdriver.Chrome] = None # Store Selenium driver if mood setter is active
        logging.info("Utilities initialized.")

    # --- REFACTOR: Keep tell_time simple, maybe return formatted string ---
    def tell_time(self) -> str:
        """
        Gets the current day, date, and time.

        Returns:
            str: A formatted string containing the current day, date, and time.
                 Example: "It is currently Wednesday, 24 July 2024 at 14:35 PM."
        """
        now = datetime.datetime.now()
        # Example format: Wednesday, 24 July 2024 at 14:35 PM
        formatted_time = now.strftime("%A, %d %B %Y at %H:%M %p")
        logging.info(f"Generated time: {formatted_time}")
        return f"It is currently {formatted_time}."

    # --- REFACTOR: Refined get_weather method ---
    def get_weather(self, location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get weather information for a location using WeatherAPI.com.

        Args:
            location (Optional[str]): City name or location query. If None, uses
                                      the default location from config.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing formatted weather information
                                      if successful, None otherwise.
        """
        target_location = location if location else self.config.DEFAULT_WEATHER_LOCATION
        api_key = self.config.WEATHER_API_KEY
        if not api_key or "YOUR_DEFAULT" in api_key:
            logging.error("WeatherAPI key not configured in config.py.")
            return None

        # Build the API URL
        url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={target_location}&days=1&aqi=no&alerts=no"
        logging.info(f"Fetching weather for '{target_location}' from {url.split('?')[0]}...") # Don't log key

        try:
            response = requests.get(url, timeout=15) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Parse the results as JSON
            jsonData = response.json()

            # Extract relevant data safely using .get()
            current = jsonData.get("current", {})
            forecast_day = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("day", {})
            location_info = jsonData.get("location", {})
            astro_info = jsonData.get("forecast", {}).get("forecastday", [{}])[0].get("astro", {})

            # Create a dictionary with desired weather information
            # --- REFACTOR: Simplified structure, clearer keys ---
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
            logging.info(f"Weather fetched successfully for '{target_location}'. Condition: {weather_info['condition']}")
            return weather_info

        # --- REFACTOR: Handle specific errors ---
        except requests.exceptions.Timeout:
            logging.error(f"Weather request timed out for location '{target_location}'.")
            return None
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error fetching weather for '{target_location}': {e.response.status_code} {e.response.reason}")
            try:
                 error_details = e.response.json()
                 logging.error(f"API Error Details: {error_details}")
            except json.JSONDecodeError:
                 logging.error(f"API Error Response (non-JSON): {e.response.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching weather for '{target_location}': {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
             logging.error(f"Error decoding weather JSON response: {e}")
             return None
        except Exception as e:
            logging.error(f"Unexpected error in get_weather for '{target_location}': {e}", exc_info=True)
            return None


    # --- REFACTOR: Mood setter (internal method) ---
    def _start_youtube_mood(self) -> bool:
        """
        (Internal) Opens a predefined YouTube video using Selenium to set the mood.

        Returns:
            bool: True if the browser was launched and video likely started, False otherwise.
        """
        if not SELENIUM_AVAILABLE:
            logging.error("Cannot start mood setter: Selenium library is not available.")
            return False
        if self._selenium_driver is not None:
             logging.warning("Mood setter already running. Stopping existing instance first.")
             self._stop_youtube_mood() # Stop previous one if exists

        url = self.config.YOUTUBE_MOOD_URL
        logging.info(f"Starting mood setter: Launching browser to {url}")

        try:
            # --- REFACTOR: Basic Selenium setup with error handling ---
            # Optional: Use webdriver-manager to automatically handle chromedriver
            # service = ChromeService(ChromeDriverManager().install())
            # options = webdriver.ChromeOptions()
            # options.add_argument("--headless") # Optional: Run headless
            # self._selenium_driver = webdriver.Chrome(service=service, options=options)

            # Basic setup (assumes chromedriver is in PATH or specified)
            options = webdriver.ChromeOptions()
            # options.add_argument("--headless") # Run headless if GUI is not desired/possible
            self._selenium_driver = webdriver.Chrome(options=options)

            self._selenium_driver.get(url)
            time.sleep(5) # Increased wait for potential ads/page load

            # Find and click the play button (robustness could be improved)
            # YouTube's class names can change. Consider more robust selectors (e.g., CSS selector with aria-label).
            play_button_selector = "button.ytp-play-button[aria-label*='Play']" # More specific selector
            try:
                 play_button = self._selenium_driver.find_element(By.CSS_SELECTOR, play_button_selector)
                 play_button.click()
                 logging.info("Clicked YouTube play button.")
                 return True
            except NoSuchElementException:
                 logging.warning("Could not find YouTube play button using selector. Video might auto-play or UI changed.")
                 # Assume it might be playing anyway? Or return False? Let's assume success for now.
                 return True
            except WebDriverException as click_err:
                 logging.error(f"Error clicking YouTube play button: {click_err}")
                 return False # Clicking failed

        except WebDriverException as e:
            logging.error(f"Selenium WebDriver error starting mood setter: {e}", exc_info=True)
            # Common issue: chromedriver version mismatch or not found in PATH
            logging.error("Ensure chromedriver is installed and compatible with your Chrome version, or use webdriver-manager.")
            self._selenium_driver = None # Ensure driver is None on failure
            return False
        except Exception as e:
             logging.error(f"Unexpected error starting mood setter: {e}", exc_info=True)
             if self._selenium_driver:
                  self._selenium_driver.quit()
             self._selenium_driver = None
             return False

    def _stop_youtube_mood(self) -> bool:
        """(Internal) Quits the Selenium browser instance if it's running."""
        if self._selenium_driver:
            logging.info("Stopping mood setter: Quitting browser instance.")
            try:
                self._selenium_driver.quit()
                self._selenium_driver = None
                return True
            except WebDriverException as e:
                 logging.error(f"Error quitting Selenium driver: {e}")
                 self._selenium_driver = None # Reset even on error
                 return False
            except Exception as e:
                 logging.error(f"Unexpected error stopping mood setter: {e}", exc_info=True)
                 self._selenium_driver = None
                 return False
        else:
             logging.info("Mood setter browser not running.")
             return True # Nothing to stop




    # --- REFACTOR: Central service dispatcher ---
    async def dispatch_service(self, ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Parses the AI response for function triggers and executes the corresponding service.

        Args:
            ai_response (str): The response text from the AI, potentially containing triggers.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the service name and its result
                                      (e.g., weather data, time string, success/failure bool).
                                      Returns None if no trigger is found or execution fails.
                                      Example: {"service": "GET_WEATHER", "result": weather_dict}
                                               {"service": "START_VIDEO", "result": True}
                                               {"service": "TELL_TIME", "result": "It is..."}
        """
        if not isinstance(ai_response, str):
             return None

        match = re.search(rf"{TRIGGER_PREFIX}(\w+)(?::(.*))?", ai_response)
        if not match:
            # logging.debug("No function trigger found in AI response.") # Can be noisy
            return None

        service_name = match.group(1).upper() # Extract service name (uppercase)
        parameter = match.group(2) if match.group(2) else None # Extract optional parameter
        logging.info(f"Detected trigger: Service='{service_name}', Parameter='{parameter}'")

        result_payload: Dict[str, Any] = {"service": service_name, "result": None}
        service_executed = False

        try:
            if service_name == SERVICE_GET_WEATHER:
                # Parameter is the location
                weather_data = self.get_weather(location=parameter)
                result_payload["result"] = weather_data
                service_executed = True
            elif service_name == SERVICE_TELL_TIME:
                time_string = self.tell_time()
                result_payload["result"] = time_string
                service_executed = True
            elif service_name == SERVICE_SET_MOOD:
                success = self._start_youtube_mood()
                result_payload["result"] = success
                service_executed = True
                # NOTE: Stopping the mood (closing browser) needs a separate trigger or logic
            elif service_name == SERVICE_START_VIDEO:
                success = await self._execute_start_video()
                result_payload["result"] = success
                service_executed = True
            elif service_name == SERVICE_STOP_VIDEO:
                success = await self._execute_stop_video()
                result_payload["result"] = success
                service_executed = True
            # --- Add other service handlers here ---
            # elif service_name == SERVICE_PLAY_SPOTIFY:
            #     # Call your spotify function with parameter
            #     pass
            else:
                logging.warning(f"Unknown service trigger name: {service_name}")
                return None # Unknown service

            if service_executed:
                 logging.info(f"Service '{service_name}' executed. Result: {str(result_payload['result'])[:100]}...")
                 return result_payload
            else:
                 # This case shouldn't be reached if all handlers set service_executed
                 logging.warning(f"Service handler for '{service_name}' did not explicitly mark execution.")
                 return None

        except Exception as e:
             logging.error(f"Unexpected error during dispatch for service '{service_name}': {e}", exc_info=True)
             return None # Return None on unexpected error during dispatch/execution


    def cleanup(self):
        """Cleans up resources used by Utilities, like the Selenium driver."""
        logging.info("Cleaning up Utilities resources...")
        self._stop_youtube_mood() # Ensure browser is closed
        logging.info("Utilities cleanup finished.")


# --- REFACTOR: Removed original monitor_sypher, choose_service, weather_service, switch_mode* ---