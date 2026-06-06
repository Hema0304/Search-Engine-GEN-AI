from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

def get_tools():
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

    return [wiki, arxiv]