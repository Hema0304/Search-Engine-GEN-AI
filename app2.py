import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun

from langchain.agents import initialize_agent, Tool, AgentType

# -----------------------
# ENV
# -----------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("🔎 AI Search Assistant (Groq + Tools)")

# -----------------------
# API KEY
# -----------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.stop()

# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-8b-8192"
)

# -----------------------
# TOOLS
# -----------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for real-time web search"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for general knowledge from Wikipedia"
    ),
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="Useful for scientific research papers"
    )
]

# -----------------------
# AGENT (STABLE VERSION)
# -----------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# -----------------------
# CHAT MEMORY
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        result = agent.run(query)

        st.session_state.messages.append(
            {"role": "assistant", "content": result}
        )

        st.write(result)
