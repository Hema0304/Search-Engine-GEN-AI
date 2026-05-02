# #AI Search Assistant using Agent + Tools (LangChain + Groq)
# AI decides:“Do I know this?”, “Should I use a tool?”
# Uses tools like:Web Search,Wikipedia, Arxiv Papers





import streamlit as st
 
from langchain_groq import ChatGroq

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  #Wrap external APIs into usable format
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)
#WikipediaQueryRun → fetch Wikipedia info
# ArxivQueryRun → fetch research papers
# DuckDuckGoSearchRun → web search

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq



llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama3-8b-8192"
)

from langchain.agents import AgentExecutor, create_tool_calling_agent
#Core of decision-making system


from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
#Shows step-by-step reasoning in UI

from dotenv import load_dotenv

load_dotenv()

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]


st.title("AI Search Assistant (Groq + LangChain)")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web. Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        streaming=False
    )

    # Prompt (MANDATORY in new API)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a smart assistant that uses tools when needed."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")  #Internal memory for agent Stores:Tool calls,Intermediate reasoning
    ])

    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())

        response = agent_executor.invoke(
            {"input": prompt},
            {"callbacks": [st_cb]}
        )

        output = response["output"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": output
        })

        st.write(output)
