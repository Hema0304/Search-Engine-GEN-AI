from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from tools import get_tools
from langchain.memory import ConversationBufferMemory
from callbacks import ToolTrackerCallback

def build_agent(groq_api_key: str):

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    tools = get_tools()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    tracker = ToolTrackerCallback()

    agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

    return agent, tracker
# from langchain_groq import ChatGroq
# from langchain.agents import initialize_agent, AgentType
# from tools import get_tools
# from langchain.memory import ConversationBufferMemory

# def build_agent(groq_api_key: str):
    
#     # 1. Load LLM dynamically using user key
#     llm = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name="llama-3.1-8b-instant"
#     )

#     # 2. Load tools
#     tools = get_tools()

#     # 3. Memory (conversation aware)
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )

#     # 4. Create agent
#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#         memory=memory,
#         verbose=True
#     )

#     return agent