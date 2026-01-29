# 🧠 Intelligent Wikipedia + Web Retrieval Agent  
## 🔗 **LangGraph-Orchestrated RAG Agent with Wikipedia & Live Web Search**

---

## 📌 **Overview**

This project implements an **intelligent, state-driven question-answering agent** built on **LangGraph**, combining **Retrieval-Augmented Generation (RAG)** with **conditional web search**.

The agent uses **LangGraph as the core orchestration engine** to manage reasoning, retrieval, and tool execution as an **explicit graph-based workflow**.  
It first queries a **Wikipedia-based vector store** and dynamically decides whether to invoke an **external web search (Tavily)** when internal knowledge is insufficient.

This design ensures **controlled reasoning**, **deterministic routing**, and **source-grounded answers**.

---

## ✨ **Key Features (LangGraph-First)**

### 🔗 **LangGraph-Driven Agent Architecture**
Explicit graph-based control over agent reasoning, retrieval, and tool delegation.

### 🧠 **State-Driven Decision Making**
The agent tracks:
- **Message history**
- **Retrieved context**
- **Tool usage state**

This prevents **infinite loops** and **hallucinated tool calls**.

### 🔍 **Hybrid Retrieval-Augmented Generation (RAG)**
Combines **local Wikipedia knowledge** with **live web search** only when needed.

### 📚 **Wikipedia Vector Knowledge Base**
- Wikipedia pages → **text chunks**
- Embedded using **SentenceTransformers**
- Stored in **ChromaDB** for semantic retrieval

### 🌐 **Conditional Web Search via Tavily API**
Web search is invoked by the **LangGraph reasoning node**, not directly by the LLM.

### ✍️ **Policy-Driven Prompt Engineering**
Structured prompts enforce:
- **FINAL ANSWER** generation
- Explicit **TOOL CALL** signaling
- **Deterministic routing decisions**

### 🤗 **Hugging Face LLM Integration**
Context-aware generation using **open-source LLMs**.

---

## 🏗️ **LangGraph Architecture**
### 🔄 **Graph-Based Agent Flow**


```text
User Question
      │
      ▼
┌──────────────────────┐
│ Wikipedia Retriever  │  ← Semantic search (ChromaDB)
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│ Reasoning Node (LLM) │  ← Decide: answer or search
└──────────────────────┘
      │
      ├── FINAL ANSWER ───────────────▶ END
      │
      ▼
┌──────────────────────┐
│ Web Search (Tavily)  │  ← Live external retrieval
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│ Reasoning Node (LLM) │  ← Synthesize final answer
└──────────────────────┘
      │
      ▼
     END
```

**LangGraph explicitly controls every transition** between retrieval, reasoning, and tool execution.

---

## 🧩 **Graph Nodes Explained**

### 🔍 **Retriever Node (Internal RAG)**
- Performs semantic search over **Wikipedia embeddings**
- Injects **top-k results** into the agent state
- **No LLM usage** → fast and deterministic

### 🧠 **Reasoning Node (LLM)**
- Evaluates retrieved context
- Decides whether information is sufficient
- Emits either:
  - **FINAL ANSWER**
  - or **TOOL CALL: `<query>`**

### 🌐 **Tool Executor Node (Tavily)**
- Executes external web search
- Injects live results back into agent context
- **Tool usage is tracked in state** to avoid repetition

---

## 🧪 **Evaluation & Benchmarking**

- Integrated with **GAIA benchmark questions**
- Supports **batch agent execution**
- Each answer is traceable to:
  - **Wikipedia RAG**
  - **Tavily web search**
  - Or **both**

---

## 🛠️ **Tech Stack**

### 🔤 **Language**
- **Python**

### 📦 **Frameworks & Libraries**
- **LangGraph** (agent orchestration)
- **LangChain** (documents, embeddings)
- **ChromaDB** (vector store)
- **SentenceTransformers**
- **Wikipedia-API**
- **Tavily API**
- **Hugging Face Hub**
- **smolagents**

### 🧠 **Core Concepts**
- **LangGraph agent workflows**
- **Retrieval-Augmented Generation (RAG)**
- **State-driven reasoning**
- **Tool delegation & orchestration**
- **Prompt engineering for LLM control**

---

## 🚀 **Why LangGraph?**

This project highlights why **LangGraph is superior to linear chains** for agent systems:

- **Explicit control flow**
- **Deterministic behavior**
- **Safe tool usage**
- **Scalable multi-node reasoning**
- **Production-ready agent design**
