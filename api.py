from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import run_user_query, user_states

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

class ClearHistoryRequest(BaseModel):
    user_id: str

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    response_message = run_user_query(request.user_id, request.input)
    return response_message

@app.post("/clear_history")
async def clear_history(request: ClearHistoryRequest, background_tasks: BackgroundTasks):
    user_id = request.user_id
    if user_id in user_states:
        user_states[user_id]["chat_history"] = []
    return {"message": "Chat history cleared."}
