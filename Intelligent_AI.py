# Fetch evaluation questions from Hugging Face Agents Course API
import requests

BASE_URL = "https://agents-course-unit4-scoring.hf.space"

# Fetch all evaluation questions
resp = requests.get(f"{BASE_URL}/questions")
questions = resp.json()

print("Number of questions:", len(questions))
print("Example question:")
print(questions[0])


# Fetch a random evaluation question from the API
import requests

BASE_URL = "https://agents-course-unit4-scoring.hf.space"

resp = requests.get(f"{BASE_URL}/random-question")
random_question = resp.json()

# Print the random question
print("Random Question:", random_question)


# Install required libraries
# pip install requests wikipedia-api langchain langchain-core langgraph smolagents huggingface-hub chromadb sentence-transformers
#!pip install langchain-core langchain-community langgraph chromadb sentence-transformers wikipedia-api smolagents

# Imports
import os
import requests
import wikipediaapi
from typing import TypedDict

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from smolagents import InferenceClientModel
from huggingface_hub import InferenceClient


#Main LLM
os.environ["HF_TOKEN"] = ""

client = InferenceClient(token=os.environ["HF_TOKEN"])
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

messages = [
    {"role": "system", "content": "You are a friendly AI."},
    {"role": "user", "content": "Hello, how are you today?"}
]

output = client.chat_completion(
    model=model_id,
    messages=messages,
    max_tokens=256
)

print(output.choices[0].message["content"])





