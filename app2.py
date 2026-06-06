import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()

st.set_page_config(page_title="AI Search Assistant", layout="wide")
st.title("🔎 AI Search Assistant (Groq + Tools)")

# ----------------------------
# API KEY
# ----------------------------
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Please enter Groq API key")
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
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

search_tool = DuckDuckGoSearchRun(name="web_search")

tools = [search_tool, wiki_tool, arxiv_tool]

# ----------------------------
# PROMPT (IMPORTANT FOR TOOL AGENT)
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a smart AI assistant. "
     "Use tools when required:\n"
     "- Web search for latest info\n"
     "- Wikipedia for concepts\n"
     "- Arxiv for research papers"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ----------------------------
# AGENT
# ----------------------------
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ----------------------------
# CHAT MEMORY
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search web, Wikipedia, and Arxiv papers."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------
# USER INPUT
# ----------------------------
query = st.chat_input("Ask anything...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())

        response = agent_executor.invoke(
            {"input": query},
            {"callbacks": [st_cb]}
        )

        output = response["output"]

        st.session_state.messages.append(
            {"role": "assistant", "content": output}
        )

        st.write(output)
