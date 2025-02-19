from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage

from Agent.main_agent import main_agent_pipeline
from Tools.tools_innit import tool_str_to_func


class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: List[AgentAction]


def execute_step(state: AgentState) -> AgentState:
    """
    Крок виклику LLM (основний агент).
    Отримуємо від LLM інструкцію, який інструмент викликати (або final_answer).
    """
    # Викликаємо пайплайн
    out = main_agent_pipeline.invoke(state)

    # Якщо LLM не вказав жодного інструмента — завершуємо (final_answer)
    if not out.tool_calls:
        final_answer = out.content
        new_action = AgentAction(
            tool="final_answer",
            tool_input={"answer": final_answer},
            log=final_answer
        )
        state["intermediate_steps"].append(new_action)
        return state

    # Якщо є виклики інструментів - беремо перший
    call = out.tool_calls[0]
    tool_name = call["name"]
    tool_args = call["args"]

    new_action = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    state["intermediate_steps"].append(new_action)

    return state


def execute_tool_step(state: AgentState) -> AgentState:
    """
    Крок для виклику конкретного інструмента (останній у списку intermediate_steps).
    Викликаємо реальний тул, логуємо його виконання та результат.
    """
    last_action = state["intermediate_steps"][-1]
    tool_name = last_action.tool
    tool_input = last_action.tool_input

    # Виводимо лог про виклик
    print(f"Executing tool '{tool_name}' with args: {tool_input}")

    # Запускаємо тул
    result = tool_str_to_func[tool_name].invoke(input=tool_input)

    # Записуємо результат
    updated_action = AgentAction(
        tool=tool_name,
        tool_input=tool_input,
        log=str(result)
    )
    state["intermediate_steps"][-1] = updated_action

    # Виводимо лог про результат
    print(f"Tool executed '{tool_name}' result: {result}")
    # Відокремлюємо рискою
    print("-" * 200)

    return state


def decide_next_node(state: AgentState) -> str:
    """
    Визначаємо, в який вузол перейти після execute_step.
    Якщо останній tool=final_answer -> йдемо у final_answer.
    Якщо інший tool -> виконуємо вузол інструмента.
    """
    if not state["intermediate_steps"]:
        return "final_answer"

    last_tool = state["intermediate_steps"][-1].tool

    if last_tool in ["holiday_info_tool", "product_lookup_tool", "shop_info_tool", "sql_db_tool", "final_answer"]:
        return last_tool

    return "final_answer"


# ----------------------
# Побудова графа
# ----------------------
graph = StateGraph(AgentState)

graph.add_node("main_agent", execute_step)
graph.add_node("holiday_info_tool", execute_tool_step)
graph.add_node("product_lookup_tool", execute_tool_step)
graph.add_node("shop_info_tool", execute_tool_step)
graph.add_node("sql_db_tool", execute_tool_step)
graph.add_node("final_answer", execute_tool_step)

graph.set_entry_point("main_agent")
graph.add_conditional_edges(source="main_agent", path=decide_next_node)

for node in ["holiday_info_tool", "shop_info_tool", "sql_db_tool"]:
    graph.add_edge(node, "main_agent")


graph.add_edge("product_lookup_tool", END)
graph.add_edge("final_answer", END)


if __name__ == "__main__":
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    app = graph.compile()

    # Get the graph image as a PNG
    graph_png = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    # Display the image
    display(Image(graph_png))

    # Save the image to a file
    with open("graph_image.png", "wb") as f:
        f.write(graph_png)
