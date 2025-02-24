from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from graph import graph

# Глобальний словник для зберігання стану кожного користувача
user_states = {}


def get_user_state(user_id: str) -> dict:
    """
    Повертає стан для користувача з user_id. Якщо стану немає, ініціалізуємо його.
    """
    if user_id not in user_states:
        user_states[user_id] = {
            "chat_history": [],
            "intermediate_steps": [],
            "input": ""
        }
    return user_states[user_id]


def extract_final_answer(state: dict) -> dict[str, Any] | dict[str, str | Any] | str:
    """
    Пошук останньої відповіді final_answer або product_lookup_tool у intermediate_steps.
    """
    for step in reversed(state.get("intermediate_steps", [])):
        if step.tool == "final":
            return {"response": step.tool_input.get("answer", "No answer found")}
        if step.tool == "product_lookup_tool":
            return {"response":' ', "items": step.log or "No answer found"}
    return "I don't have a response for that."



def run_user_query(user_id: str, user_input: str) -> str:
    """
    Обробляє запит користувача з урахуванням його унікального id.
    """
    # Отримуємо або ініціалізуємо стан для даного користувача
    state = get_user_state(user_id)

    # Додаємо запит користувача до стану
    state["input"] = user_input

    # Компіляція та виклик графа з поточним станом
    compiled_graph = graph.compile()
    state = compiled_graph.invoke(state)

    # Отримуємо відповідь від системи
    response = extract_final_answer(state)

    # Оновлюємо історію чату користувача
    state["chat_history"].append(HumanMessage(content=user_input))
    state["chat_history"].append(AIMessage(content=response["response"]))

    # Очищаємо "input" та накопичені кроки для наступного запиту
    state["input"] = ""
    state["intermediate_steps"] = []

    # Зберігаємо оновлений стан користувача
    user_states[user_id] = state

    return response


# Для тестування з консолі
if __name__ == "__main__":
    test_user_id = "test_user"
    print("Ласкаво просимо до чату. Напишіть ваше повідомлення.")
    while True:
        inp = input("User: ")
        reply = run_user_query(test_user_id, inp)
        print("Assistant:", reply)
