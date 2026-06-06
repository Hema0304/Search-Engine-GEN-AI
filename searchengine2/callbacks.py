from langchain.callbacks.base import BaseCallbackHandler

class ToolTrackerCallback(BaseCallbackHandler):

    def __init__(self):
        self.tools_used = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        self.tools_used.append(tool_name)

    def get_tools(self):
        return self.tools_used