from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import process_message, chat_histories

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    input: str

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    response_message = process_message(request.user_id, request.input)
    return {"response": response_message}

@app.post("/clear_history")
async def clear_history(request: ChatRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id in chat_histories:
        chat_histories[user_id] = []
    return {"message": "Chat history cleared."}