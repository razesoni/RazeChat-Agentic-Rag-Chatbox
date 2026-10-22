import os 
import math
import sqlite3
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from src.tools import tools

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
)

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Nodes 1
def chat_node(state: ChatState):
    """
    LLM node that can answer directly or call an appropriate tool.
    """

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `purchase_stock` when the user wants to purchase a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# Node 2 - tool node
tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        configurable = ckpt.config.get('configurable')
        if configurable is not None:
            thread_id = configurable.get('thread_id')
            if thread_id is not None:
                all_threads.add(thread_id)

    return list(all_threads)