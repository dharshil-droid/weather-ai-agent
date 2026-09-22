
import os
import requests
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


# ==========================================
# API KEYS
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")


# ==========================================
# WEATHER FUNCTION
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
        return {
            "error": "Could not find weather information for this city."
        }

    data = response.json()

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]
    wind_speed = data["wind"]["speed"]

    return {
        "city": city,
        "temperature": temperature,
        "feels_like": feels_like,
        "humidity": humidity,
        "description": description,
        "wind_speed": wind_speed
    }


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def weather_tool(city: str):
    """Get current weather information for a city."""
    return get_weather(city)


# ==========================================
# GEMINI MODEL
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY
)


# ==========================================
# AI AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=[weather_tool],

    system_prompt="""
You are a specialized Weather AI Agent.

You ONLY answer questions related to weather and climate
of places.

You can answer questions about:
- Current weather
- Temperature
- Humidity
- Rain
- Wind
- Weather conditions
- Weather forecasts
- Weather comparisons between places
- Basic climate information

When the user asks about current weather,
always use the weather_tool.

If the question is unrelated to weather or climate,
respond:

"Sorry, I can only answer questions related to weather and climate."

Never make up current weather information.

Keep answers simple and easy to understand.
"""
)


# ==========================================
# CHAT FUNCTION
# ==========================================

def chat(user_message):

    if not user_message.strip():
        return "Please enter a weather-related question."

    try:

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        })

        return result["messages"][-1].content

    except Exception as e:

        return f"Sorry, something went wrong: {str(e)}"


# ==========================================
# GRADIO INTERFACE
# ==========================================

demo = gr.Interface(

    fn=chat,

    inputs=gr.Textbox(
        label="Ask about weather",
        placeholder="Example: What is the weather in Hyderabad?"
    ),

    outputs=gr.Markdown(
        label="Weather Agent"
    ),

    title="🌦️ Weather AI Agent",

    description=(
        "Ask me about the weather or climate of places. "
        "I only answer weather and climate related questions."
    ),

    examples=[
        "What is the weather in Hyderabad?",
        "Is it raining in Delhi?",
        "What is the temperature in Mumbai?",
        "What is the weather like in Chennai?"
    ]
)


# ==========================================
# START APP
# ==========================================

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
