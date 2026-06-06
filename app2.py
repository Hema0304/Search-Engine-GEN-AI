import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ----------------------------
# ENV
# ----------------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant")
st.title("🔎 AI Search Assistant (Stable Version)")

# ----------------------------
# API KEY
# ----------------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.stop()

# ----------------------------
# LLM
# ----------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192"
)

# ----------------------------
# TOOLS
# ----------------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
search = DuckDuckGoSearchRun()

tools = [
    Tool(name="Web Search", func=search.run, description="Use for latest info"),
    Tool(name="Wikipedia", func=wiki.run, description="Use for definitions and concepts"),
    Tool(name="Arxiv", func=arxiv.run, description="Use for research papers")
]

# ----------------------------
# AGENT (STABLE)
# ----------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# ----------------------------
# CHAT MEMORY
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------
# INPUT
# ----------------------------
query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response = agent.run(query)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.write(response)
