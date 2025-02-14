from datetime import datetime, timezone

from Agent.main_agent import main_agent_prompt
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder

# Compile the graph
runnable = graph_builder.compile()

# Chat history storage
chat_histories = {}
last_activity = {}

def process_message(user_id: str, user_input: str) -> str:
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        last_activity[user_id] = datetime.now(timezone.utc)

    chat_histories[user_id].append({"role": "user", "content": user_input})

    input_format = {
        "history": chat_histories[user_id],
        "input": user_input
    }

    response = runnable.invoke({"messages": main_agent_prompt.format_messages(**input_format)})
    response_message = response["messages"][-1].content

    chat_histories[user_id].append({"role": "assistant", "content": response_message})
    last_activity[user_id] = datetime.now(timezone.utc)

    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return response_message
