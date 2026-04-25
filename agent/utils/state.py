from typing import Optional, Dict, List, Tuple, Literal, Annotated, Any
from pydantic import BaseModel, Field

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from .db import neo4j_driver



class SpecFilter(BaseModel):
    value: Optional[float] = None
    op: Optional[Literal["gte", "lte", "eq"]] = None

class SpecState(BaseModel):
    ram: SpecFilter = Field(default_factory=SpecFilter)
    storage: SpecFilter = Field(default_factory=SpecFilter)
    screen_size: SpecFilter = Field(default_factory=SpecFilter)
    refresh_rate: SpecFilter = Field(default_factory=SpecFilter)
    response_time: SpecFilter = Field(default_factory=SpecFilter)
    weight: SpecFilter = Field(default_factory=SpecFilter)
    battery: SpecFilter = Field(default_factory=SpecFilter)
    release_year: SpecFilter = Field(default_factory=SpecFilter)

class ResolvedProduct(BaseModel):
    input_name: str
    matched_name: Optional[str] = None
    product_id: Optional[str] = None
    score: Optional[float] = None

class AgentState(BaseModel):
    user_id: Optional[str] = None

    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list
    )

    category: Optional[
        Literal[
            "tivi",
            "tablet",
            "mobile",
            "micro-thu-am",
            "may-in",
            "man-hinh",
            "laptop",
        ]
    ] = None
    brand: Optional[str] = None
    series: Optional[str] = None
    name: Optional[str] = None
    color: Optional[str] = None

    price_min: Optional[int] = None
    price_max: Optional[int] = None

    spec: SpecState = Field(default_factory=SpecState)

    mentioned_products: List[str] = Field(default_factory=list)
    resolved_products: List[ResolvedProduct] = Field(default_factory=list)



def get_user_preferences(
    customer_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get most frequent brand and color
    """

    top_brand = None
    top_color = None

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (c:Customer {id: $customer_id})-[r:BOUGHT]->(v:Variant)
            MATCH (p:Product)-[:HAS_VARIANT]->(v)
            MATCH (p)-[:OF_BRAND]->(b:Brand)
            RETURN b.name AS brand, SUM(r.amount) AS cnt
            ORDER BY cnt DESC
            LIMIT 1
            """,
            {"customer_id": customer_id},
        )
        record = result.single()
        if record:
            top_brand = record.get("brand")

    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (c:Customer {id: $customer_id})-[r:BOUGHT]->(v:Variant)
            RETURN v.color AS color, SUM(r.amount) AS cnt
            ORDER BY cnt DESC
            LIMIT 1
            """,
            {"customer_id": customer_id},
        )
        record = result.single()
        if record:
            top_color = record.get("color")

    return top_brand, top_color

def create_initial_state(
    user_id: Optional[str] = None,
    system_message: Optional[BaseMessage] = None,
) -> AgentState:
    messages = [system_message] if system_message is not None else []

    if not user_id:
        return AgentState(messages=messages)
    
    try:
        brand, color = get_user_preferences(user_id)

        return AgentState(
            user_id=user_id,
            messages=messages,
            brand=brand,
            color=color,
        )
    except Exception as e:
        print(f"[NEO4J PROFILE ERROR] user_id={user_id} error={e}")
        return AgentState(messages=messages)
