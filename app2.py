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

from langchain.agents import AgentExecutor, create_tool_calling_agent
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
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
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
