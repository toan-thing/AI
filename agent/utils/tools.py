from typing import Optional, Dict, Any, Literal, List, Annotated

import chromadb
from pydantic import BaseModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from .db import get_pg_conn, release_pg_conn, product_collection, policy_collection
from .state import AgentState



@tool
def query_products(state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """
    Search and recommend products from database based on user constraints.

    Use this tool when the user is looking for products with filters such as:
    - category, brand, series, name
    - color
    - price range
    - specifications (RAM, storage, screen size, refresh rate, response time, weight, battery, release year.)

    The tool returns top matching products sorted by average_rating * rated_quantity,
    filtered to only include products with stock available, along with full details:
    variants (price, color, stock) and technical specifications.

    Also returns total number of matched products to indicate result coverage.
    """

    if not state.category:
        return {
            "total": 0,
            "products": [],
            "note": "Category is required for product search. Ask the user to specify a product category first."
        }

    conn = get_pg_conn()
    cur = conn.cursor()

    try:
        where_clauses: List[str] = []
        values: List[Any] = []

        # Chỉ lấy sản phẩm còn hàng
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM variant v
                WHERE v.ps_product_id = p.ps_product_id
                AND v.stock > 0
            )
        """)

        if state.category:
            where_clauses.append("p.category = %s")
            values.append(state.category)

        if state.brand:
            where_clauses.append("p.brand ILIKE %s")
            values.append(f"%{state.brand}%")

        if state.series:
            where_clauses.append("p.series ILIKE %s")
            values.append(f"%{state.series}%")

        if state.color:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM variant v
                    WHERE v.ps_product_id = p.ps_product_id
                    AND v.color ILIKE %s
                )
            """)
            values.append(f"%{state.color}%")

        if state.price_min is not None:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM variant v
                    WHERE v.ps_product_id = p.ps_product_id
                    AND v.price >= %s
                )
            """)
            values.append(state.price_min)

        if state.price_max is not None:
            where_clauses.append("""
                EXISTS (
                    SELECT 1 FROM variant v
                    WHERE v.ps_product_id = p.ps_product_id
                    AND v.price <= %s
                )
            """)
            values.append(state.price_max)

        if state.spec:
            for key, cond in state.spec.model_dump().items():
                if not cond or not isinstance(cond, dict) or cond.get("value") is None:
                    continue

                op = cond.get("op")
                value = cond.get("value")

                if op == "gte":
                    where_clauses.append("""
                        EXISTS (
                            SELECT 1 FROM product_spec ps
                            WHERE ps.ps_product_id = p.ps_product_id
                            AND ps.spec_key = %s
                            AND ps.value_num >= %s
                        )
                    """)
                elif op == "lte":
                    where_clauses.append("""
                        EXISTS (
                            SELECT 1 FROM product_spec ps
                            WHERE ps.ps_product_id = p.ps_product_id
                            AND ps.spec_key = %s
                            AND ps.value_num <= %s
                        )
                    """)
                elif op == "eq":
                    where_clauses.append("""
                        EXISTS (
                            SELECT 1 FROM product_spec ps
                            WHERE ps.ps_product_id = p.ps_product_id
                            AND ps.spec_key = %s
                            AND ps.value_num = %s
                        )
                    """)
                else:
                    continue

                values.append(key)
                values.append(value)

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        id_query = f"""
            SELECT p.ps_product_id
            FROM product p
            {where_sql}
            ORDER BY (p.average_rating * p.rated_quantity) DESC NULLS LAST
            LIMIT 5
        """

        cur.execute(id_query, values)
        id_rows = cur.fetchall()
        product_ids = [r[0] for r in id_rows]

        if not product_ids:
            return {
                "total": 0,
                "products": []
            }

        count_query = f"SELECT COUNT(*) FROM product p {where_sql}"
        cur.execute(count_query, values)
        total = cur.fetchone()[0]

        hydrate_query = """
            SELECT
                p.ps_product_id,
                p.name,
                p.category,
                p.brand,
                p.series,
                p.average_rating,
                p.rated_quantity,

                json_agg(DISTINCT jsonb_build_object(
                    'id', v.ps_variant_id,
                    'color', v.color,
                    'price', v.price,
                    'stock', v.stock,
                    'image', v.image
                )) AS variants,

                json_agg(DISTINCT jsonb_build_object(
                    'key', ps.spec_key,
                    'value_num', ps.value_num,
                    'value_text', ps.value_text,
                    'unit', ps.unit
                )) AS specs

            FROM product p
            LEFT JOIN variant v ON p.ps_product_id = v.ps_product_id
            LEFT JOIN product_spec ps ON p.ps_product_id = ps.ps_product_id
            WHERE p.ps_product_id = ANY(%s::uuid[])
            GROUP BY p.ps_product_id
            ORDER BY (p.average_rating * p.rated_quantity) DESC NULLS LAST
        """

        cur.execute(hydrate_query, (product_ids,))
        rows = cur.fetchall()

        products = [
            {
                "product_id": r[0],
                "name": r[1],
                "category": r[2],
                "brand": r[3],
                "series": r[4],
                "average_rating": r[5],
                "rated_quantity": r[6],
                "variants": r[7] or [],
                "specs": r[8] or [],
            }
            for r in rows
        ]

        return {
            "total": total,
            "returned": len(products),
            "products": products,
            "note": "Top products by average_rating * rated_quantity after filtering constraints. Only products with stock > 0 are included."
        }

    finally:
        cur.close()
        release_pg_conn(conn)



@tool
def query_resolved_products(state: Annotated[AgentState, InjectedState]) -> Dict[str, Any]:
    """
    Retrieve full product details for products already identified in conversation.

    Use this tool when the user mentions specific product names and they have been
    resolved to product IDs in the agent state.

    This tool fetches complete information including:
    - product info (name, category, brand, series, average_rating, rated_quantity)
    - variants (price, color, stock, image)
    - technical specifications

    Useful for product comparison, detailed explanation, or follow-up questions.
    """

    conn = get_pg_conn()
    cur = conn.cursor()

    try:
        product_ids = list(set([
            p.product_id
            for p in state.resolved_products
            if p.product_id is not None
        ]))

        if not product_ids:
            return {
                "total": 0,
                "products": [],
                "note": "No resolved products found"
            }

        query = """
            SELECT
                p.ps_product_id,
                p.name,
                p.category,
                p.brand,
                p.series,
                p.average_rating,
                p.rated_quantity,

                json_agg(DISTINCT jsonb_build_object(
                    'id', v.ps_variant_id,
                    'color', v.color,
                    'price', v.price,
                    'stock', v.stock,
                    'image', v.image
                )) AS variants,

                json_agg(DISTINCT jsonb_build_object(
                    'key', ps.spec_key,
                    'value_num', ps.value_num,
                    'value_text', ps.value_text,
                    'unit', ps.unit
                )) AS specs

            FROM product p
            LEFT JOIN variant v ON p.ps_product_id = v.ps_product_id
            LEFT JOIN product_spec ps ON p.ps_product_id = ps.ps_product_id
            WHERE p.ps_product_id = ANY(%s::uuid[])
            GROUP BY p.ps_product_id
            ORDER BY (p.average_rating * p.rated_quantity) DESC NULLS LAST
        """

        cur.execute(query, (product_ids,))
        rows = cur.fetchall()

        products = [
            {
                "product_id": r[0],
                "name": r[1],
                "category": r[2],
                "brand": r[3],
                "series": r[4],
                "average_rating": r[5],
                "rated_quantity": r[6],
                "variants": r[7] or [],
                "specs": r[8] or [],
            }
            for r in rows
        ]

        return {
            "total": len(product_ids),
            "returned": len(products),
            "products": products,
            "note": "Hydrated from resolved_products in AgentState"
        }

    finally:
        cur.close()
        release_pg_conn(conn)



@tool
def semantic_search(
    query: str,
    collection_name: Literal["products", "policies"],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Perform semantic search over knowledge base using embeddings.

    Use this tool when the user asks about:
    - description, FAQs about specific products
    - general policies (warranty, return, shipping)

    Supports two collections:
    - "products": product descriptions, FAQs
    - "policies": store policies

    Returns relevant text chunks to support natural language answers.
    """

    k = min(max(k, 1), 10)

    if collection_name == "products":
        collection = product_collection
    else:
        collection = policy_collection

    results = collection.query(
        query_texts=[query],
        n_results=k,
    )
    docs = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    formatted = [
        {
            "text": doc,
            "distance": distances[i] if i < len(distances) else None,
        }
        for i, doc in enumerate(docs)
    ]

    return {
        "query": query,
        "collection": collection_name,
        "count": len(formatted),
        "results": formatted,
    }





def handle_tool_error(state) -> dict:
    error = state.get("error", "Unknown error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Tool error: {repr(error)}",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list):
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
    )

tools = [
    query_products,
    query_resolved_products,
    semantic_search,
]

tool_node = create_tool_node_with_fallback(tools)