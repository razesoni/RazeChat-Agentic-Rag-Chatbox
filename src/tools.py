import os
import requests
from typing import Any
import math
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig

DB_ROOT = "faiss_db"

def _db_path(thread_id: str) -> str:
    return os.path.join(DB_ROOT, thread_id)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def ingest_rag_document(file_path: str, thread_id: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    path = _db_path(thread_id)
    if os.path.exists(path):
        # Second PDF in the same chat: add to the existing index
        store = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)
    store.save_local(path)


def get_retriever(thread_id: str):
    path = _db_path(thread_id)
    if not os.path.exists(path):
        return None  # nothing uploaded for this conversation yet

    store = FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# rag tool
@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve relevant information from the PDF uploaded in this conversation.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the uploaded PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = get_retriever(thread_id) if thread_id else None

    if retriever is None:
        return "No PDF has been uploaded in this conversation. Ask the user to upload one."

    documents = retriever.invoke(query)

# Search Tool
@tool
def search_tool(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return titles, URLs, and snippets."""
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, max_results=max_results)
            )
            out = []
            for r in results:
                out.append(
                    f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body', '')[:300]}\n"
                )
            if not out:
                return "No search results found."
            return "\n----\n".join(out)
    except Exception as e:
        return f"Error performing search: {e}"

@tool
def calculator(expression: str) -> str:
    """
    Useful for basic math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("STOCK_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantitiy of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interupt
    and wait for a human verification ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} share of {symbol}.",
            "symbol": symbol,
            "quantity": quantity
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity
        }

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK" or "New York, US".
    Returns:
        A formatted current weather report.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing."
            "Set the OPENWEATHER_API_KEY environment variable."
        )
    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }
        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }
        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"

# Make tool list
tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock]