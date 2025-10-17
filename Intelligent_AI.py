# Install required libraries
# pip install requests wikipedia-api langchain langchain-core langgraph smolagents huggingface-hub chromadb sentence-transformers
#!pip install langchain-core langchain-community langgraph chromadb sentence-transformers wikipedia-api smolagents

# Imports
import os
import requests
import wikipediaapi

import re
from typing import TypedDict, List
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
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


# Build a  Wikipedia-based knowledge base
import wikipediaapi
from langchain.schema import Document

# Initialize Wikipedia API client
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='GAIA-Agent'  
)

# List of Wikipedia topics to include in the knowledge base
topics = [
    "Mercedes Sosa discography", "1928 Summer Olympics", "Ada Lovelace", "Moon landing",
    "Olympic Games medal table", "List of studio albums by Mercedes Sosa", "FIFA World Cup winners",
    "List of countries by population", "List of Nobel laureates", "Marie Curie",
    "Albert Einstein", "Isaac Newton", "Apollo 11", "First man in space",
    "Solar System", "Periodic table", "Human anatomy", "COVID-19 pandemic",
    "United Nations", "History of the Internet", "Artificial intelligence"
]

# Fetch Wikipedia content and convert to LangChain Document objects
docs = []
for topic in topics:
    page = wiki_wiki.page(topic)
    if page.exists():
        for para in page.text.split("\n"):
            if para.strip():
                docs.append(Document(page_content=para.strip(), metadata={"topic": topic}))

print(f"Collected {len(docs)} Wikipedia documents for GAIA knowledge base.")


# Build a retriever agent using Chroma and SentenceTransformers
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Create vector store from Wikipedia documents
vectorstore = Chroma.from_documents(
    docs,
    embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Convert vector store to a retriever for semantic search
retriever = vectorstore.as_retriever()


# -------------------------------
# CONFIGURATION Tavily
# -------------------------------

# Example: os.environ["TAVILY_API_KEY"] = "api_key_here"
os.environ["TAVILY_API_KEY"] = "API-KEY"

# Initialize Tavily client
try:
    if "TAVILY_API_KEY" in os.environ and os.environ["TAVILY_API_KEY"].startswith("tvly-"):
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        print("Tavily Client Initialized.")
    else:
        tavily = None
        print("Tavily API Key not properly configured.")
except Exception as e:
    tavily = None
    print(f"Error initializing Tavily: {e}")


# -------------------------------
# DEFINE LANGGRAPH STATE
# -------------------------------
class AgentState(TypedDict):
    messages: List[AnyMessage]
    context: str
    tool_used: bool  # Track if Tavily search already used


# -------------------------------
# RETRIEVER NODE (Internal Search)
# -------------------------------
def retriever_node(state: AgentState) -> AgentState:
    global retriever
    print("NODE: Running RAG (Internal Retrieval)")

    question = state["messages"][-1].content
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception as e:
        print(f"Retriever failed: {e}")
        docs = []

    context = "\n\n".join([doc.page_content for doc in docs[:3]]) if docs else ""
    return {"context": context, "tool_used": state.get("tool_used", False)}


# -------------------------------
# REASONING NODE (Decision + LLM)
# -------------------------------
def reasoning_node(state: AgentState) -> AgentState:
    global model
    print("NODE: Running Reasoning and Routing")

    question = state["messages"][-1].content
    context = state.get("context", "")

    truncated_context = context[:1000]  # avoid long LLM prompts

    full_prompt = (
        f"You are a helpful agent. Your goal is to answer the user's question.\n"
        f"1. If CONTEXT below is enough, respond with 'FINAL ANSWER: ...'\n"
        f"2. If CONTEXT is empty and no tool used, respond with 'TOOL CALL: <query>'.\n"
        f"3. If context contains Tavily info, give a FINAL ANSWER.\n\n"
        f"CONTEXT:\n{truncated_context}\n\nQUESTION: {question}\nAnswer:"
    )

    huggingface_messages = [{"role": "user", "content": full_prompt}]

    try:
        model_output = model(huggingface_messages)
    except Exception as e:
        print(f"LLM call failed: {e}")
        model_output = [{"generated_text": "TOOL CALL: " + question}]

    if isinstance(model_output, list) and model_output and isinstance(model_output[0], dict):
        ai_message_content = model_output[0].get("generated_text", "No generated text found.")
    elif isinstance(model_output, dict):
        ai_message_content = model_output.get("generated_text", "No generated text found.")
    else:
        ai_message_content = str(model_output)

    ai_message = AIMessage(content=ai_message_content)
    new_messages = state["messages"] + [ai_message]

    return {"messages": new_messages, "context": context, "tool_used": state.get("tool_used", False)}


# -------------------------------
# TOOL EXECUTOR NODE (Tavily Search)
# -------------------------------
def tool_executor_node(state: AgentState) -> AgentState:
    global tavily
    print("NODE: Executing Tavily Web Search")

    if not tavily:
        return {"context": "Error: Tavily not initialized.", "tool_used": True}

    ai_message = state["messages"][-1].content
    match = re.search(r"TOOL CALL:\s*(.*)", ai_message, re.IGNORECASE | re.DOTALL)
    if not match:
        print("TOOL NODE ERROR: Could not parse query.")
        return {"context": "Error parsing tool call.", "tool_used": True}

    query = match.group(1).strip()
    print(f"Tavily Search Query: '{query}'")

    try:
        response = tavily.search(query=query, search_depth="basic", max_results=5)
        search_results = [
            f"Source: {r['url']}\nContent: {r['content']}"
            for r in response['results']
        ]
        new_context = f"--- External Info from Tavily ({query}) ---\n" + "\n---\n".join(search_results)
    except Exception as e:
        print(f"Tavily search failed: {e}")
        new_context = f"Error: Tavily search failed. {e}"

    return {"context": new_context, "tool_used": True}


# -------------------------------
# ROUTING LOGIC
# -------------------------------
def route_to_tool_or_end(state: AgentState) -> str:
    llm_output = state["messages"][-1].content.upper()
    tool_used = state.get("tool_used", False)

    if "FINAL ANSWER" in llm_output:
        return END
    elif "TOOL CALL" in llm_output and not tool_used:
        return "tool_executor"
    else:
        return END


# -------------------------------
# BUILD AND COMPILE GRAPH
# -------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("retriever", retriever_node)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "reasoning")
workflow.add_conditional_edges(
    "reasoning",
    route_to_tool_or_end,
    {"tool_executor": "tool_executor", END: END}
)
workflow.add_edge("tool_executor", "reasoning")

app = workflow.compile()
print("Agent Graph Compiled Successfully with Tavily integration.")



# Fetch GAIA questions
BASE_URL = "https://agents-course-unit4-scoring.hf.space"
resp = requests.get(f"{BASE_URL}/questions")
questions = resp.json()
print(f"Fetched {len(questions)} GAIA questions.")
print("Example question:", questions[0])



#EXECUTION BLOCK FOR GAIA QUESTIONS

answers = []
for q in questions:
    task_id = q["task_id"]
    question_text = q["question"]

    # Initialize agent state
    state_input = {
        "messages": [HumanMessage(content=question_text)],
        "context": "",
        "tool_used": False
    }

    # Invoke the agent
    result = app.invoke(state_input)

    # Extract final message
    final_message_content = result["messages"][-1].content

    # Determine source
    context_used = result.get("context", "")
    if "--- External Information from Tavily Web Search" in context_used:
        source = "Tool (Tavily Search)"
    elif final_message_content.startswith("FINAL ANSWER"):
        source = "Wikipedia RAG"
    else:
        source = "Unknown"

    # Print GAIA question, answer, and source
    print("\n" + "="*80)
    print(f"GAIA Question [{task_id}]: {question_text}")
    print("Final Answer:", final_message_content)
    print("Source:", source)

    # Save for submission
    answers.append({
        "task_id": task_id,
        "submitted_answer": final_message_content,
        "source": source
    })

print("\nAll GAIA answers processed and ready for submission.")





