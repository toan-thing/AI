from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Optional, Dict
import time
import os
import traceback

from langchain_core.messages import HumanMessage, messages_from_dict

from agent.agent import graph
from agent.utils.state import create_initial_state, AgentState
from agent.utils.db import neo4j_driver

from dotenv import load_dotenv
load_dotenv()





app = FastAPI()
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
    product_name: str
    color: str
    price: float
    amount: int = 1

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        cleanup_sessions()

        state = get_or_create_session(
            req.session_id,
            req.customer_id
        )

        state_dict = state.model_dump()
        state_dict["messages"] = state.messages + [HumanMessage(content=req.message)]
        result_dict = graph.invoke(state_dict)

        updated_state = AgentState(**result_dict)
        save_session(req.session_id, updated_state)

        reply = ""
        if updated_state.messages:
            reply = updated_state.messages[-1].content

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
                MATCH (p:Product {name: $product_name})-[:HAS_VARIANT]->(v:Variant)
                WHERE v.color = $color AND v.price = $price
                RETURN v
            """, {
                "product_name": req.product_name,
                "color": req.color,
                "price": req.price,
            })

            record = result.single()
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail=f"No variant found for product '{req.product_name}' with color '{req.color}' and price {req.price}"
                )

            session.run("""
                MERGE (c:Customer {id: $customer_id})
                WITH c
                MATCH (p:Product {name: $product_name})-[:HAS_VARIANT]->(v:Variant)
                WHERE v.color = $color AND v.price = $price
                MERGE (c)-[r:BOUGHT]->(v)
                ON CREATE SET r.amount = $amount
                ON MATCH SET r.amount = r.amount + $amount
            """, {
                "customer_id": req.customer_id,
                "product_name": req.product_name,
                "color": req.color,
                "price": req.price,
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
