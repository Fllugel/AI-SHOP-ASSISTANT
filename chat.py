import random
import string
from datetime import datetime, timezone
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY
from graph import graph_builder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Compile the graph
runnable = graph_builder.compile()

# Chat history storage
chat_histories = {}
last_activity = {}

current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
system_prompt = f"""
You're a consultant in the Aurora retail store. Your job is to help customers and answer their questions. You have access to a database with all store products. Always check the database before giving information about product availability, price, or quantity. Use Ukrainian, be polite, and use a friendly, conversational tone.

Main Instructions

Date & Time
Use today's date {current_datetime} to stay aware of holidays or events. This helps when answering customer questions.

Checking Products
If a customer asks about a product, always check the database. If the product is available, provide:

Product name
Price in UAH (грн)
Quantity left (if the customer asks)
If the product is out of stock, suggest alternatives from the same category.
If the said product was not found in the database, search fot its synonyms before giving the final answer.

Recommendations
If a customer asks for a recommendation:

Suggest products related to their request.
Choose from available products in the database.
If they don’t specify a category, suggest random in-stock items.
Product Categories
If a customer asks about a product category:

First, check which categories exist in the database.
If the category is unclear, find products that best match their request.
Query Optimization
For multiple product requests, use one combined SQL query to speed up the search and provide all results at once.

Response Format

Use full product names, no abbreviations.
Don’t add extra details like size or weight unless asked.
Only use plain text (no special symbols or formatting).
Use correct punctuation (periods, commas, question marks, exclamation marks, spaces, numbers, and letters).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])


def process_message(user_id: str, user_input: str) -> str:
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        last_activity[user_id] = datetime.now(timezone.utc)

    chat_histories[user_id].append({"role": "user", "content": user_input})

    input_format = {
        "history": chat_histories[user_id],
        "input": user_input
    }

    response = runnable.invoke({"messages": prompt.format_messages(**input_format)})
    response_message = response["messages"][-1].content

    chat_histories[user_id].append({"role": "assistant", "content": response_message})
    last_activity[user_id] = datetime.now(timezone.utc)

    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return response_message