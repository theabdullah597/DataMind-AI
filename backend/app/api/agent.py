import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.agent.graph import process_chat

router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    dataset_id: str = None

@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        response_text = await asyncio.to_thread(process_chat, messages_dict, request.dataset_id)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
