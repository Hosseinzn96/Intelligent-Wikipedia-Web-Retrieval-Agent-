Intelligent Wikipedia + Web Retrieval Agent
(RAG with LangChain & LangGraph)


#Overview
This project implements an intelligent agent for open-domain question answering by combining Retrieval-Augmented Generation (RAG) with external web search.
The system retrieves relevant information from a Wikipedia-based knowledge base and, when necessary, performs real-time web searches to provide accurate, source-backed answers.


#Key Features

1) Retrieval-Augmented Generation (RAG) combining internal knowledge with live web search.

2) Wikipedia Knowledge Base built using SentenceTransformers and stored in ChromaDB.

3) Agent Workflow with LangGraph, featuring custom nodes for:
Internal retrieval
Reasoning and decision routing
External search via Tavily API

4) Prompt Engineering techniques to optimize LLM reasoning and response accuracy.

5) Hugging Face LLM Integration for context-aware, source-backed generation.

#Architecture

Data Collection: Wikipedia pages are processed into document chunks and embedded using SentenceTransformers.

Storage: Documents are stored in ChromaDB for fast semantic search.

Reasoning Graph: LangGraph manages the flow between retrieval, reasoning, and external search nodes.

External Search: When needed, Tavily API fetches live web results.

Prompt Engineering: Custom prompts guide LLM behavior and ensure structured, factual outputs.

#Tech Stack
Languages & Frameworks: Python

Libraries: LangChain, LangGraph, ChromaDB, SentenceTransformers, Tavily, Hugging Face

Concepts: RAG, LLM Reasoning, Prompt Engineering, Agent Design
