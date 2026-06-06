def route_query(query: str):
    q = query.lower()

    if "paper" in q or "arxiv" in q or "research" in q:
        return "arxiv"

    return "wikipedia"