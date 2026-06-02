from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
import time
import os
import traceback
import threading
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage, AIMessage
from consumers.worker import start_kafka_consumer


from agent.agent import graph
from agent.utils.state import create_initial_state, AgentState
from agent.utils.db import neo4j_driver, close_all

from dotenv import load_dotenv
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Starting Kafka Consumer Thread ---")
    kafka_thread = threading.Thread(target=start_kafka_consumer, daemon=True)
    kafka_thread.start()
    
    yield
    print("--- Shutting down AI Service ---")
    close_all()


app = FastAPI(
    title="AI Agent Service",
    openapi_url="/api/agent/openapi.json", 
    docs_url="/api/agent/docs",            
    redoc_url=None,
    lifespan=lifespan                         
)
router = APIRouter(prefix="/api/agent")

class SessionData:
    def __init__(self, state: AgentState):
        self.state = state
        self.last_updated = time.time()

sessions: Dict[str, SessionData] = {}

SESSION_TTL = 60 * 30

def get_or_create_session(session_id: str, customer_id: Optional[str]) -> AgentState:
    if session_id in sessions:
        sessions[session_id].last_updated = time.time()
        return sessions[session_id].state

    state = create_initial_state(user_id=customer_id)
    sessions[session_id] = SessionData(state)
    return state

def save_session(session_id: str, state: AgentState):
    sessions[session_id] = SessionData(state)

def cleanup_sessions():
    now = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if now - data.last_updated > SESSION_TTL
    ]
    for sid in expired:
        del sessions[sid]



class ChatRequest(BaseModel):
    session_id: str
    message: str
    customer_id: Optional[str] = None

class PurchaseRequest(BaseModel):
    customer_id: str
    variant_id: str
    amount: int = 1

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        cleanup_sessions()

        state = get_or_create_session(
            req.session_id,
            req.customer_id
        )

        result_dict = graph.invoke({
            **state.model_dump(exclude={"messages"}),
            "messages": state.messages + [HumanMessage(content=req.message)],
        })

        updated_state = AgentState(**result_dict)
        save_session(req.session_id, updated_state)

        reply = next((m.content for m in reversed(updated_state.messages) if isinstance(m, AIMessage)),"")

        return {
            "session_id": req.session_id,
            "reply": reply
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/purchase")
def update_purchase(req: PurchaseRequest):
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (v:Variant {id: $variant_id})
                RETURN v
            """, {"variant_id": req.variant_id})

            record = result.single()
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail=f"No variant found with id '{req.variant_id}'"
                )

            session.run("""
                MERGE (c:Customer {id: $customer_id})
                WITH c
                MATCH (v:Variant {id: $variant_id})
                MERGE (c)-[r:BOUGHT]->(v)
                ON CREATE SET r.amount = $amount
                ON MATCH SET r.amount = r.amount + $amount
            """, {
                "customer_id": req.customer_id,
                "variant_id": req.variant_id,
                "amount": req.amount,
            })

        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id].state

    return {
        "user_id": state.user_id,
        "category": state.category,
        "brand": state.brand,
        "series": state.series,
        "color": state.color,
        "price_min": state.price_min,
        "price_max": state.price_max,
        "spec": state.spec.model_dump(),
        "mentioned_products": state.mentioned_products,
        "resolved_products": [
            p.model_dump() for p in state.resolved_products
        ],
        "messages_count": len(state.messages)
    }

@router.post("/reset/{session_id}")
def reset(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "reset"}

@router.get("/health")
def health():
    return {
        "status": "ok",
        "sessions": len(sessions)
    }



app.include_router(router)

