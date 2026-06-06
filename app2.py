import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("🔎 AI Search Assistant (Groq + Tools)")

# -----------------------------
# API KEY
# -----------------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Please enter Groq API Key")
    st.stop()

# -----------------------------
# TOOLS
# -----------------------------
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)

wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)
search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Web Search",
        func=search_tool.run,
        description="Use for latest news, general questions, or unknown queries"
    ),
    Tool(
        name="Wikipedia",
        func=wiki_tool.run,
        description="Use for definitions, history, concepts, people, places"
    ),
    Tool(
        name="Arxiv",
        func=arxiv_tool.run,
        description="Use for AI/ML research papers and scientific topics"
    )
]

# -----------------------------
# LLM (Groq)
# -----------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192",
    streaming=True
)

# -----------------------------
# AGENT (STABLE VERSION)
# -----------------------------
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------------
# SESSION MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search web, Wikipedia, and research papers. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response = agent_executor.run(query)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.write(response)
