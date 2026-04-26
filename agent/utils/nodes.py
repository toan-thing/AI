from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from .state import AgentState, SpecState, SpecFilter, ResolvedProduct
from .tools import tools
from .db import get_pg_conn, release_pg_conn





def build_llm_input(state: AgentState) -> Dict[str, Any]:
    messages = state.messages[-10:]

    resolved_products = [
        {
            "product_id": p.product_id,
            "name": p.matched_name,
            "score": p.score,
        }
        for p in state.resolved_products
    ]

    return {
        "messages": messages,
        "state": {
            "category": state.category,
            "brand": state.brand,
            "series": state.series,
            "color": state.color,

            "price_min": state.price_min,
            "price_max": state.price_max,

            "spec": state.spec.model_dump(),

            "mentioned_products": state.mentioned_products,
            "resolved_products": resolved_products,
        },
    }



class ParseOutput(BaseModel):
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
    color: Optional[str] = None

    price_min: Optional[int] = None
    price_max: Optional[int] = None

    spec: SpecState = Field(default_factory=SpecState)

    mentioned_products: List[str] = Field(default_factory=list)

class Parse:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        result: ParseOutput = self.runnable.invoke({
            "messages": state.messages[-10:],
        })

        updates = {}

        if (
            result.category is not None
            and state.category is not None
            and result.category != state.category
        ):
            updates = {
                "category": result.category,
                "brand": None,
                "series": None,
                "color": None,
                "price_min": None,
                "price_max": None,
                "spec": SpecState(),
                "mentioned_products": [],
                "resolved_products": [],
            }

        if result.category is not None:
            updates["category"] = result.category

        if result.brand and result.brand.strip():
            updates["brand"] = result.brand.strip()

        if result.series and result.series.strip():
            updates["series"] = result.series.strip()

        if result.color and result.color.strip():
            updates["color"] = result.color.strip()

        if result.price_min is not None:
            updates["price_min"] = result.price_min

        if result.price_max is not None:
            updates["price_max"] = result.price_max

        spec_updates = {}
        for field in state.spec.model_fields.keys():
            src_value: SpecFilter = getattr(result.spec, field)

            if not src_value:
                continue

            if src_value.value is None and src_value.op is None:
                continue

            spec_updates[field] = src_value

        if spec_updates:
            base_spec = updates.get("spec", state.spec)
            updates["spec"] = base_spec.model_copy(update=spec_updates, deep=True)

        if result.mentioned_products:
            seen = set()
            merged = []
            
            existing = updates.get("mentioned_products", state.mentioned_products)

            for name in existing + result.mentioned_products:
                if not isinstance(name, str):
                    continue
                clean = name.strip()
                key = clean.lower()
                if not clean or key in seen:
                    continue
                seen.add(key)
                merged.append(clean)

            updates["mentioned_products"] = merged

        return updates

parse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Bạn là hệ thống trích xuất thông tin sản phẩm từ câu người dùng.

🎯 MỤC TIÊU:
- Trích xuất thông tin CHÍNH XÁC từ input
- KHÔNG suy đoán
- CHỈ điền field khi chắc chắn

--------------------------------------------------

📦 CÁC FIELD:

- category: một trong [
    "tivi", "tablet", "mobile", "micro-thu-am",
    "may-in", "man-hinh", "laptop"
]

- brand: hãng (vd: Dell, Apple, Asus)
- series: dòng sản phẩm (vd: ThinkPad, MacBook Pro)
- color: màu sắc

- price_min: giá tối thiểu (int)
- price_max: giá tối đa (int)

- spec: object gồm:
    ram, storage, screen_size, refresh_rate,
    response_time, weight, battery, release_year

Mỗi spec phải có dạng:
{{
  "value": number,
  "op": "gte" | "lte" | "eq"
}}

- mentioned_products: danh sách tên sản phẩm người dùng nhắc đến

--------------------------------------------------

⚠️ QUY TẮC QUAN TRỌNG:

1. Nếu KHÔNG có thông tin → để null (hoặc object/list rỗng)

2. KHÔNG đoán:
   ❌ "laptop tốt" → KHÔNG suy ra brand
   ❌ "giá rẻ" → KHÔNG tự đặt price

3. Chuẩn hóa:
   - brand, series, color → string sạch (trim)
   - mentioned_products → giữ nguyên tên

4. spec:
   - CHỈ tạo khi có số cụ thể
   - KHÔNG tạo nếu thiếu value hoặc op

5. category:
   - CHỈ chọn từ danh sách
   - nếu không chắc → null

--------------------------------------------------

📌 FORMAT OUTPUT (JSON):

{{
  "category": "laptop",
  "brand": "Apple",
  "series": "MacBook Pro",
  "color": null,
  "price_min": 20000000,
  "price_max": null,
  "spec": {{
    "ram": {{"value": 16, "op": "gte"}},
    "storage": {{"value": 512, "op": "gte"}}
  }},
  "mentioned_products": ["MacBook Pro M3"]
}}

--------------------------------------------------

🚫 KHÔNG:
- Không viết text ngoài JSON
- Không giải thích
- Không thêm field ngoài schema

👉 CHỈ trả về JSON hợp lệ
"""
    ),
    ("placeholder", "{messages}")
])

# parse_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.0,
#     max_output_tokens=512,
# )
parse_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
)

parse_runnable = parse_prompt | parse_llm.with_structured_output(ParseOutput)



def Resolve_products(state: AgentState, config: RunnableConfig):
    if not state.mentioned_products:
        return {}

    conn = get_pg_conn()
    cur = conn.cursor()

    try:
        existing_ids = {
            p.product_id
            for p in state.resolved_products
            if isinstance(p, ResolvedProduct) and p.product_id is not None
        }

        resolved: List[ResolvedProduct] = list(state.resolved_products)

        for raw_name in state.mentioned_products:
            if not isinstance(raw_name, str):
                continue

            clean_name = raw_name.strip()
            if not clean_name:
                continue

            cur.execute(
                """
                SELECT id, name, rating
                FROM product
                WHERE name ILIKE %s
                ORDER BY rating DESC
                LIMIT 1
                """,
                (f"%{clean_name}%",),
            )

            row = cur.fetchone()
            if not row:
                continue

            product_id, matched_name, rating = row

            if product_id in existing_ids:
                continue

            resolved.append(
                ResolvedProduct(
                    input_name=clean_name,
                    matched_name=matched_name,
                    product_id=product_id,
                    score=float(rating) if rating is not None else None,
                )
            )

            existing_ids.add(product_id)

        return {
            "resolved_products": resolved,
            "mentioned_products": [],
        }

    finally:
        cur.close()
        release_pg_conn(conn)



class Reason:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: AgentState, config: RunnableConfig):
        llm_input = build_llm_input(state)
        local_messages = list(llm_input["messages"])
        while True:
            result = self.runnable.invoke({**llm_input, "messages": local_messages})
            if (not result.content or (
                isinstance(result.content, list)
                and (not result.content or not result.content[0].get("text"))
            )) and not result.tool_calls:
                local_messages.append(SystemMessage(content="Hãy trả lời lại cho khách hàng một cách rõ ràng và cụ thể."))
            else:
                break
        return {"messages": result}

reason_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Bạn là AI tư vấn sản phẩm (laptop, điện thoại, TV, màn hình...).

🎯 MỤC TIÊU:
- Hiểu nhu cầu người dùng từ hội thoại
- Sử dụng state (filter, spec, resolved_products)
- Quyết định: hỏi thêm / gọi tool / trả lời

--------------------------------------------------

🧠 NGỮ CẢNH (STATE):
Bạn có thể truy cập:
- category, brand, series, color
- price_min, price_max
- spec (ram, storage, screen_size, ...)
- mentioned_products
- resolved_products (đã map sang product_id)

--------------------------------------------------

🚦 QUY TRÌNH QUYẾT ĐỊNH (RẤT QUAN TRỌNG):

Bước 1 — Kiểm tra yêu cầu người dùng:

A. Nếu người dùng:
- hỏi chung chung (vd: "laptop nào tốt?")
- thiếu thông tin quan trọng (giá, nhu cầu, loại sản phẩm)

→ HÀNH ĐỘNG:
- KHÔNG gọi tool
- Hỏi lại để làm rõ (1–2 câu ngắn)

--------------------------------------------------

B. Nếu người dùng:
- đã cung cấp đủ thông tin cơ bản, ví dụ:
    - có category (laptop, mobile…)
    - có ít nhất 1 trong các yếu tố:
        + giá (price_min / price_max)
        + spec (ram, storage…)
        + brand / series

→ HÀNH ĐỘNG:
- GỌI TOOL để tìm sản phẩm
- KHÔNG trả lời bằng kiến thức tự suy đoán

--------------------------------------------------

C. Nếu:
- đã có kết quả từ tool
- hoặc đã có resolved_products

→ HÀNH ĐỘNG:
- KHÔNG gọi tool nữa
- Trả lời trực tiếp:
    + gợi ý sản phẩm
    + so sánh
    + giải thích ngắn gọn

--------------------------------------------------

D. Nếu người dùng hỏi về sản phẩm cụ thể (vd: "MacBook Pro M3 có tốt không?")
→ HÀNH ĐỘNG:
- Nếu đã resolve được → trả lời luôn
- Nếu chưa → có thể gọi tool để lấy thêm thông tin

--------------------------------------------------

📌 NGUYÊN TẮC GỌI TOOL:

CHỈ gọi tool khi:
✔ Có category rõ ràng
✔ Và có ít nhất 1 filter (price / spec / brand)

KHÔNG gọi tool khi:
✘ Thiếu thông tin quan trọng
✘ Câu hỏi còn mơ hồ
✘ Đã có dữ liệu đủ để trả lời

--------------------------------------------------

📌 ƯU TIÊN:

1. Hiểu đúng nhu cầu
2. Hỏi nếu thiếu
3. Gọi tool khi đủ
4. Trả lời khi có dữ liệu

--------------------------------------------------

📌 VÍ DỤ:

User: "laptop tốt"
→ hỏi lại (KHÔNG gọi tool)

User: "laptop gaming dưới 30 triệu"
→ gọi tool

User: "MacBook Pro M3 có đáng mua không"
→ trả lời luôn (không cần tool nếu đã biết)

User: "so sánh 2 sản phẩm X và Y"
→ nếu đã resolve → trả lời, không gọi tool

--------------------------------------------------

⚠️ QUY TẮC QUAN TRỌNG:

- KHÔNG bịa dữ liệu sản phẩm
- KHÔNG đoán nếu thiếu thông tin
- KHÔNG gọi tool nếu chưa cần
- KHÔNG trả lời lan man

- ƯU TIÊN:
    1. Hiểu đúng nhu cầu
    2. Gọi tool khi cần
    3. Trả lời rõ ràng

--------------------------------------------------

🗣️ STYLE TRẢ LỜI:

- Ngắn gọn (2–5 câu)
- Tự nhiên, giống tư vấn viên
- Không dùng markdown phức tạp
- Không giải thích hệ thống

--------------------------------------------------

⏰ Thời gian hiện tại: {time}

👉 Hãy quyết định hành động tiếp theo (hỏi / gọi tool / trả lời)
"""
    ),
    ("system", """
STATE:
{state}
"""),
    ("placeholder", "{messages}")
]).partial(time=datetime.now)

# reason_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.2,
#     max_output_tokens=768,
#     timeout=20,
#     max_retries=3,
# )
reason_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    timeout=20,
    max_retries=3,
)

reason_runnable = reason_prompt | reason_llm.bind_tools(
    tools,
)
