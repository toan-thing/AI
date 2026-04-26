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
Bạn là hệ thống trích xuất thông tin sản phẩm từ hội thoại người dùng.

═══════════════════════════════════════════════════════════

📦 CÁC FIELD CẦN TRÍCH XUẤT:

category — chỉ chọn từ danh sách này, không chọn ngoài:
  "tivi" | "tablet" | "mobile" | "micro-thu-am" | "may-in" | "man-hinh" | "laptop"

brand       — tên hãng (vd: Dell, Apple, Samsung)
series      — dòng sản phẩm (vd: ThinkPad, MacBook Pro, Galaxy S)
color       — màu sắc (vd: đen, trắng, xanh)
price_min   — giá tối thiểu (số nguyên, đơn vị VNĐ)
price_max   — giá tối đa (số nguyên, đơn vị VNĐ)

spec — gồm các trường: ram, storage, screen_size, refresh_rate,
                        response_time, weight, battery, release_year

  Mỗi spec có dạng: {{"value": <số>, "op": <toán tử>}}

mentioned_products — danh sách tên sản phẩm cụ thể người dùng nhắc đến

═══════════════════════════════════════════════════════════

🔢 QUY TẮC TOÁN TỬ SPEC — CỰC KỲ QUAN TRỌNG:

Chỉ được dùng đúng 3 giá trị sau cho "op", TUYỆT ĐỐI không dùng giá trị nào khác:

  "gte" → lớn hơn hoặc bằng
  "lte" → nhỏ hơn hoặc bằng
  "eq"  → bằng chính xác

Ánh xạ từ tiếng Việt:
  "hơn", "trên", "tối thiểu", "ít nhất", "từ X trở lên"  → "gte"
  "dưới", "tối đa", "không quá", "nhỏ hơn", "đến X"      → "lte"
  "đúng X", "chính xác X"                                 → "eq"

❌ TUYỆT ĐỐI KHÔNG dùng: "gt", "lt", "greater", "less", ">=", "<=" hay bất kỳ dạng nào khác.

═══════════════════════════════════════════════════════════

📏 QUY TẮC TRÍCH XUẤT:

1. CHỈ điền field khi người dùng nói RÕ RÀNG — không suy đoán, không phỏng đoán.

2. Không suy diễn:
   ❌ "laptop tốt"    → KHÔNG suy ra brand hay price
   ❌ "giá rẻ"        → KHÔNG tự đặt price_min / price_max
   ❌ "pin trâu"      → KHÔNG tự đặt giá trị battery

3. Giá tiền:
   - "20 triệu" = 20000000
   - "dưới 20 triệu" → price_max = 20000000
   - "từ 15 đến 25 triệu" → price_min = 15000000, price_max = 25000000

4. Spec — CHỈ tạo khi có số cụ thể VÀ có thể xác định được op:
   ✅ "RAM 16GB"          → ram: {{"value": 16, "op": "gte"}}
   ✅ "pin hơn 4000 mAh"  → battery: {{"value": 4000, "op": "gte"}}
   ✅ "màn 144Hz"         → refresh_rate: {{"value": 144, "op": "gte"}}
   ❌ "pin tốt"           → KHÔNG tạo (không có số)
   ❌ "RAM cao"           → KHÔNG tạo (không có số)

5. mentioned_products — tên sản phẩm cụ thể người dùng nhắc đến (giữ nguyên tên):
   ✅ "iPhone 15 Pro Max", "MacBook Pro M3", "Galaxy S24 Ultra"
   ❌ Tên chung như "iPhone", "laptop Dell" không phải sản phẩm cụ thể

6. Nếu thiếu thông tin → để null (hoặc list/object rỗng).

═══════════════════════════════════════════════════════════

📌 VÍ DỤ OUTPUT:

Input: "tư vấn laptop Dell RAM 16GB dưới 25 triệu màu đen"
Output:
{{
  "category": "laptop",
  "brand": "Dell",
  "series": null,
  "color": "đen",
  "price_min": null,
  "price_max": 25000000,
  "spec": {{
    "ram": {{"value": 16, "op": "gte"}}
  }},
  "mentioned_products": []
}}

Input: "so sánh iPhone 15 Pro và Samsung Galaxy S24"
Output:
{{
  "category": "mobile",
  "brand": null,
  "series": null,
  "color": null,
  "price_min": null,
  "price_max": null,
  "spec": {{}},
  "mentioned_products": ["iPhone 15 Pro", "Samsung Galaxy S24"]
}}

Input: "laptop tốt nhất hiện nay"
Output:
{{
  "category": "laptop",
  "brand": null,
  "series": null,
  "color": null,
  "price_min": null,
  "price_max": null,
  "spec": {{}},
  "mentioned_products": []
}}

═══════════════════════════════════════════════════════════

🚫 TUYỆT ĐỐI KHÔNG:
- Viết text ngoài JSON
- Thêm field ngoài schema
- Dùng op ngoài "gte" / "lte" / "eq"
- Suy đoán thông tin không có trong input

👉 Chỉ trả về JSON hợp lệ, không giải thích.
"""
    ),
    ("placeholder", "{messages}")
])

parse_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
)
# parse_llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.0,
# )

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

            if len(clean_name.split()) < 2:
                continue

            cur.execute(
                """
                SELECT id, name, rating,
                    similarity(name, %s) AS sim
                FROM product
                WHERE similarity(name, %s) > 0.3
                ORDER BY sim DESC, rating DESC
                LIMIT 1
                """,
                (clean_name, clean_name),
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
Bạn là nhân viên tư vấn bán hàng AI của một cửa hàng điện tử (laptop, điện thoại, TV, màn hình, máy tính bảng...).

═══════════════════════════════════════════════════════════

🧠 THÔNG TIN BẠN CÓ (STATE):

- category       — loại sản phẩm
- brand          — hãng sản xuất
- series         — dòng sản phẩm
- color          — màu sắc yêu cầu
- price_min / price_max — khoảng giá
- spec           — yêu cầu kỹ thuật (ram, storage, battery, ...)
- resolved_products — sản phẩm cụ thể đã được tìm thấy trong hệ thống
- mentioned_products — tên sản phẩm khách vừa nhắc đến

═══════════════════════════════════════════════════════════

🚦 QUY TRÌNH QUYẾT ĐỊNH:

── BƯỚC 1: Khách hỏi chung chung hoặc thiếu thông tin? ──

Nếu thiếu category HOẶC thiếu tất cả filter (giá, spec, brand):
→ Hỏi lại 1–2 câu ngắn để làm rõ.
→ KHÔNG gọi tool.
→ KHÔNG đoán sản phẩm.

Ví dụ cần hỏi lại:
  "laptop tốt" → hỏi ngân sách và nhu cầu
  "điện thoại pin trâu" → hỏi khoảng giá

── BƯỚC 2: Đủ thông tin để tìm kiếm? ──

Điều kiện gọi tool query_products:
  ✔ Có category
  ✔ Có ÍT NHẤT 1 trong: price / spec / brand / series

Điều kiện gọi tool query_resolved_products:
  ✔ Khách nhắc sản phẩm cụ thể theo tên
  ✔ resolved_products đã có dữ liệu

Điều kiện gọi tool semantic_search:
  ✔ Khách hỏi về chính sách (đổi trả, bảo hành, giao hàng)
  ✔ Khách hỏi mô tả / FAQ về sản phẩm cụ thể

── BƯỚC 3: Đã có kết quả? ──

Nếu đã có dữ liệu từ tool hoặc resolved_products:
→ Trả lời trực tiếp.
→ KHÔNG gọi tool thêm lần nữa.
→ KHÔNG bịa thêm thông tin ngoài dữ liệu có sẵn.

═══════════════════════════════════════════════════════════

⚠️ CÁC LỖI THƯỜNG GẶP — TUYỆT ĐỐI TRÁNH:

❌ Gọi tool khi đã có đủ dữ liệu để trả lời
❌ Trả lời bằng kiến thức tự có mà không dùng dữ liệu từ tool
❌ Bịa giá, thông số, hoặc tên sản phẩm không có trong state
❌ Gọi tool khi câu hỏi còn mơ hồ
❌ Lặp lại tool call nhiều lần với cùng tham số

═══════════════════════════════════════════════════════════

📌 VÍ DỤ QUYẾT ĐỊNH:

"laptop tốt"
→ Hỏi lại (thiếu giá và nhu cầu)

"laptop gaming dưới 30 triệu RAM 16GB"
→ Gọi query_products

"MacBook Pro M3 giá bao nhiêu?"
→ Gọi query_resolved_products (đã có trong resolved_products)
→ Hoặc trả lời trực tiếp nếu đã có dữ liệu

"chính sách đổi trả thế nào?"
→ Gọi semantic_search với collection "policies"

"so sánh iPhone 15 và Galaxy S24"
→ Nếu resolved_products đã có cả hai → trả lời luôn
→ Nếu chưa → gọi query_resolved_products

═══════════════════════════════════════════════════════════

🗣️ PHONG CÁCH TRẢ LỜI:

- Tự nhiên, thân thiện như nhân viên tư vấn thực thụ
- Ngắn gọn: 2–4 câu cho câu hỏi đơn, tối đa 6–8 câu khi so sánh/giới thiệu nhiều sản phẩm
- Trình bày sản phẩm: nêu tên, giá, 1–2 điểm nổi bật phù hợp với nhu cầu khách
- KHÔNG dùng markdown phức tạp (không dùng ###, không dùng bảng)
- KHÔNG giải thích cách hệ thống hoạt động
- KHÔNG nói "Dựa trên dữ liệu từ tool..." hay các cụm tương tự
- Nếu không có sản phẩm phù hợp → nói thật, đề nghị điều chỉnh yêu cầu

═══════════════════════════════════════════════════════════

⏰ Thời gian hiện tại: {time}
"""
    ),
    (
        "system",
        """STATE HIỆN TẠI:
{state}"""
    ),
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
