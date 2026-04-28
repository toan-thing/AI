import os
from typing import List

import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from psycopg2.pool import SimpleConnectionPool
from neo4j import GraphDatabase, Driver

from dotenv import load_dotenv
load_dotenv()



def must_getenv(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing env: {key}")
    return value



pg_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=must_getenv("PG_HOST"),
    database=must_getenv("PG_DB"),
    user=must_getenv("PG_USER"),
    password=must_getenv("PG_PASSWORD"),
    port=int(os.getenv("PG_PORT", "5432")),
)

def get_pg_conn():
    return pg_pool.getconn()

def release_pg_conn(conn):
    if conn:
        pg_pool.putconn(conn)



neo4j_driver: Driver = GraphDatabase.driver(
    must_getenv("NEO4J_URI"),
    auth=(
        must_getenv("NEO4J_USER"),
        must_getenv("NEO4J_PASSWORD"),
    ),
)



# try:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
# except Exception:
#     device = "cpu"
device = os.getenv("EMBEDDING_DEVICE", "cpu")

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="dangvantuan/vietnamese-embedding",
    device=device,
    normalize_embeddings=True,
)

chroma_client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "./vectordb")
)

product_collection = chroma_client.get_collection(
    name="products",
    embedding_function=embedding_function,
)

policy_collection = chroma_client.get_collection(
    name="policies",
    embedding_function=embedding_function,
)



def close_all():
    try:
        pg_pool.closeall()
    except Exception:
        pass

    try:
        neo4j_driver.close()
    except Exception:
        pass