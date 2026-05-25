from fastapi import HTTPException
from pydantic import BaseModel

from agent.utils.db import neo4j_driver

class PurchaseRequest(BaseModel):
    customer_id: str
    variant_id: str
    amount: int = 1

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
