import streamlit as st
from dotenv import load_dotenv
import arxiv

from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import (
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)

from langchain.tools import Tool

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

# -------------------------
# Wikipedia Tool
# -------------------------
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500
    )
)

# -------------------------
# Custom Arxiv Tool
# -------------------------
def search_arxiv(query: str) -> str:
    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []

        for paper in client.results(search):
            papers.append(
                f"""
Title: {paper.title}

Authors: {", ".join(author.name for author in paper.authors)}

Published: {paper.published.date()}

Summary: {paper.summary[:500]}

Link: {paper.entry_id}
"""
            )

        if not papers:
            return "No papers found."

        return "\n\n".join(papers)

    except Exception as e:
        return f"Arxiv Error: {str(e)}"


arxiv_tool = Tool(
    name="Arxiv",
    func=search_arxiv,
    description="Useful for searching research papers from arXiv."
)

# -------------------------
# DuckDuckGo Search Tool
# -------------------------
search = DuckDuckGoSearchRun(name="Search")

# -------------------------
# Tool List
# -------------------------
tools = [
    search,
    wiki,
    arxiv_tool
]

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI Search Assistant using Agent + Tools")
st.caption("Web Search + Wikipedia + Arxiv Papers")

api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Groq API key in the sidebar.")
    st.stop()

# -------------------------
# Chat Memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can search the web, Wikipedia, and arXiv research papers."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------
# User Query
# -------------------------
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    st.chat_message("user").write(prompt)

    # -------------------------
    # Groq LLM
    # -------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        streaming=False
    )

    # -------------------------
    # Prompt
    # -------------------------
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a smart AI assistant.

Use:
- Web Search for current information.
- Wikipedia for encyclopedic information.
- Arxiv for research papers.

Always choose the best tool when needed.
"""
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    # -------------------------
    # Agent
    # -------------------------
    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt_template
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # -------------------------
    # Response
    # -------------------------
    with st.chat_message("assistant"):

        st_cb = StreamlitCallbackHandler(st.container())

        try:
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_cb]}
            )

            output = response["output"]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": output
                }
            )

            st.write(output)

        except Exception as e:
            st.error(f"Error: {str(e)}")
