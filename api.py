from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from chat import chat_histories, last_activity, runnable, prompt
from config import MAX_MESSAGES_IN_SHORT_TERM_MEMORY

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"], #TODO: Fix api: HOW TO HANDLE HTTPS?
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    input: str

@app.post("/chat") #TODO: Fix api: Api py and chat uses repeat code
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    last_activity[user_id] = datetime.now(timezone.utc)

    # Build the input payload
    input_payload = {
        "history": chat_histories[user_id],
        "input": request.input
    }

    # Invoke the graph
    response = runnable.invoke({
        "messages": prompt.format_messages(**input_payload)
    })

    # Extract the assistant's response
    response_message = response["messages"][-1].content

    # Update chat history
    chat_histories[user_id].append({"role": "user", "content": request.input})
    chat_histories[user_id].append({"role": "assistant", "content": response_message})

    # Enforce memory limit
    if len(chat_histories[user_id]) > MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:
        chat_histories[user_id] = chat_histories[user_id][-MAX_MESSAGES_IN_SHORT_TERM_MEMORY * 2:]

    return {"response": response_message}

@app.post("/clear_history") #TODO: Fix api: Decide is this nececery and how to handle multiple users history
async def clear_history(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id in chat_histories:
        chat_histories[user_id] = []
    print(f"Clearing history for user_id: {user_id}")
    return {"message": "Chat history cleared."}