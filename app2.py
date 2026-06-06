import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# -----------------------
# ENV
# -----------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")

st.title("🔎 AI Search Assistant (Groq + Tools + Agent)")

# -----------------------
# SIDEBAR API KEY
# -----------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# -----------------------
# TOOLS SETUP
# -----------------------

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

search_tool = DuckDuckGoSearchRun(name="web_search")

tools = [search_tool, wiki_tool, arxiv_tool]

# -----------------------
# LLM (Groq)
# -----------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192",
    streaming=True
)

# -----------------------
# PROMPT
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent assistant. "
     "Use tools (web search, wikipedia, arxiv) whenever needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# -----------------------
# AGENT
# -----------------------

# -----------------------------
# 1. WRAP TOOLS (same as before)
# -----------------------------
tools = [
    Tool(
        name="Web Search",
        func=search_tool.run,
        description=(
            "Use ONLY for real-time or general questions such as: "
            "latest news, current events, trending topics, product info, tutorials, or unknown queries."
        )
    ),
    Tool(
        name="Wikipedia",
        func=wiki_tool.run,
        description=(
            "Use ONLY for factual, encyclopedic knowledge such as: "
            "history, definitions, concepts, places, people, and background explanations."
        )
    ),
    Tool(
        name="Arxiv Research",
        func=arxiv_tool.run,
        description=(
            "Use ONLY for research-level questions such as: "
            "machine learning papers, AI models, algorithms, scientific studies, and technical innovations."
        )
    )
]

# -----------------------------
# 2. SMART SYSTEM INSTRUCTION
# -----------------------------
SYSTEM_PREFIX = """
You are an advanced AI Search Assistant.

You MUST follow these rules:

1. FIRST understand the user intent:
   - If the question is about NEWS, CURRENT EVENTS → use Web Search
   - If the question is about DEFINITIONS, HISTORY, GENERAL KNOWLEDGE → use Wikipedia
   - If the question is about RESEARCH PAPERS, AI, ML, SCIENCE → use Arxiv Research

2. If you are unsure → prefer Web Search.

3. NEVER guess answers without tools when factual accuracy is needed.

4. Always give:
   - Clear explanation
   - Structured answer (bullet points if needed)
   - Simple language

5. If multiple tools are needed:
   - Combine results and summarize clearly

6. Be concise but informative.
"""

# -----------------------------
# 3. AGENT INITIALIZATION
# -----------------------------
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
    agent_kwargs={
        "prefix": SYSTEM_PREFIX
    }
)
# -----------------------
# SESSION MEMORY
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search web, Wikipedia, and research papers. Ask me anything."}
    ]

# show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------
# USER INPUT
# -----------------------
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())

        result = agent_executor.invoke(
            {"input": user_input},
            {"callbacks": [callback]}
        )

        answer = result["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.write(answer)
