import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import OPENAI_MODEL_NAME
from Tools.tools_innit import tools

load_dotenv(dotenv_path=".env")


# Define State object structure
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the graph builder
graph_builder = StateGraph(State)

# Set up the LLM with tools binding
llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.7, api_key=os.getenv("GPT_API_KEY"))
llm_with_tools = llm.bind_tools(tools=tools)


# Define the chatbot response function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tools node to the graph
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges for tool use
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define entry and exit edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("tools", "chatbot")

if __name__ == "__main__":
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    app = graph_builder.compile()

    # Get the graph image as a PNG
    graph_png = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    # Display the image
    display(Image(graph_png))

    # Save the image to a file
    with open("graph_image.png", "wb") as f:
        f.write(graph_png)
