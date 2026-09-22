
import os
import requests
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# API KEYS
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")


# ==========================================
# GEMINI MODEL
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY
)


# ==========================================
# WEATHER API
# ==========================================

def get_weather(city):

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    data = response.json()

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }


# ==========================================
# AI CHAT FUNCTION
# ==========================================

def chat(user_message):

    if not user_message.strip():
        return "Please enter a weather-related question."

    try:

        prompt = f"""
You are a weather question classifier.

User question:
{user_message}

If the question is about weather or climate,
extract the city name.

Return ONLY:

WEATHER: city_name

If the question is NOT about weather or climate,
return ONLY:

NOT_WEATHER
"""

        response = llm.invoke(prompt)

        # Handle Gemini response
        if isinstance(response.content, list):
            answer = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            ).strip()
        else:
            answer = str(response.content).strip()

        print("Gemini classification:", answer)

        # Reject unrelated questions
        if answer.upper() == "NOT_WEATHER":
            return "Sorry, I can only answer questions related to weather and climate."

        # Extract city
        if answer.upper().startswith("WEATHER:"):

            city = answer.split(":", 1)[1].strip()

            weather = get_weather(city)

            if weather is None:
                return "Sorry, I could not find weather information for that city."

            return f"""
### 🌦️ Weather in {weather['city']}

🌡️ **Temperature:** {weather['temperature']} °C

🌡️ **Feels like:** {weather['feels_like']} °C

☁️ **Condition:** {weather['description'].capitalize()}

💧 **Humidity:** {weather['humidity']}%

💨 **Wind speed:** {weather['wind_speed']} m/s
"""

        return "Sorry, I couldn't understand the question."

    except Exception as e:

        return f"Sorry, something went wrong: {str(e)}"
