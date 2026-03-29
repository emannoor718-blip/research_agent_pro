# tools.py
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def get_tools():
    search = DuckDuckGoSearchRun(name="web_search")
    arxiv  = ArxivQueryRun(name="arxiv_search")
    wiki   = WikipediaQueryRun(
        name="wikipedia_search",
        api_wrapper=WikipediaAPIWrapper(top_k_results=2)
    )
    return [search, arxiv, wiki]