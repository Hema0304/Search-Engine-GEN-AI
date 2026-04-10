import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Tools
wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
)

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
)

search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# UI
st.title("LangChain - Chat with Search (Groq)")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can search the web. Ask me anything."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True
    )

    # Prompt (REQUIRED in new API)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can use tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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





# import streamlit as st 
# from langchain_groq import ChatGroq
# from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
# from langchain_community.tools import (
#     ArxivQueryRun,
#     WikipediaQueryRun,
#     DuckDuckGoSearchRun
# )
# from langchain.agents import initialize_agent, AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# import os 
# from dotenv import load_dotenv
# load_dotenv()


# api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
# wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
# arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

# search=DuckDuckGoSearchRun(name="Search")

# st.title("Langchain - Chat with search")

# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your Groq API Key:",type="password")


# if "messages" not in st.session_state:
#     st.session_state["messages"]=[
#         {"role":"assistant","content":"Hi,I'm a chatbot who can search the web. How can i help you?"}
#     ]
    
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg['content'])
    
# if prompt:=st.chat_input(placeholder="what is machine learning?"):
#     st.session_state.messages.append({"role":"user","content":prompt})
#     st.chat_message("user").write(prompt)
    
#     llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
#     tools=[search,arxiv,wiki]
    
#     search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    
#     with st.chat_message("assistant"):
#         st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#         response = search_agent.run(prompt, callbacks=[st_cb])
#         st.session_state.messages.append({'role':'assistant',"content":response})
#         st.write(response)
    