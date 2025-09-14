**Hướng dẫn: Triển khai Agent Hỗ trợ Khách hàng bằng LangGraph và Google Gemini trên máy chủ cục bộ**

---

## Tổng quan

Trong hướng dẫn này, chúng ta sẽ xây dựng và triển khai một agent hỗ trợ khách hàng thông minh sử dụng LangGraph để điều phối luồng xử lý, và Google Gemini thông qua gói `langchain-google-genai` làm LLM nền tảng. Cuối cùng, bạn sẽ có một dịch vụ HTTP FastAPI chạy cục bộ (hoặc trong Docker) có khả năng phân loại truy vấn, phân tích cảm xúc, tạo phản hồi, đồng thời tự động chuyển tiếp khi cần.

## Yêu cầu tiên quyết

* Python 3.9 trở lên
* Một dự án Google Cloud đã kích hoạt **Gemini API** và có khóa API hoặc khóa tài khoản dịch vụ
* Kiến thức cơ bản về Python, FastAPI và Docker (tùy chọn)

## 1. Cấu trúc dự án

```text
support-agent/
├── backend/
│   ├── app.py
│   └── customer_support.py
├── frontend/
├── tests/
├── .env
├── requirements.txt
├── Dockerfile        # tùy chọn
└── README.md         # bạn đang xem ở đây
```

## 2. Thiết lập môi trường

1. **Tạo hoặc chuyển đến** thư mục dự án:

   ```bash
   mkdir support-agent && cd support-agent
   ```
2. **Tạo và kích hoạt** môi trường ảo:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

## 3. Cài đặt Thư viện

Tạo file `requirements.txt`:

```text
langgraph
langchain-core
langchain-google-genai
google-genai
fastapi
uvicorn
python-dotenv
```

Cài đặt các gói:

```bash
pip install -r requirements.txt
```

## 4. Cấu hình Khóa API

Tạo file `.env` ở gốc dự án:

```dotenv
# .env
GEMINI_API_KEY=<KHÓA_API_GEMINI_CỦA_BẠN>
GEMINI_MODEL=gemini-1.5-pro
```

Trong ví dụ này, biến `GEMINI_MODEL` được đặt mặc định là `gemini-1.5-pro`.

Cả `GEMINI_API_KEY` và `GEMINI_MODEL` đều là bắt buộc; nếu thiếu, chương trình sẽ báo lỗi `EnvironmentError`.

> **Lưu ý:** Bạn cũng có thể đặt biến `GOOGLE_API_KEY`, gói `langchain-google-genai` sẽ tự động lấy giá trị này.

## 5. Định nghĩa Workflow (`backend/customer_support.py`)

```python
# backend/customer_support.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Tải khóa Gemini từ .env
load_dotenv()
# Tùy chọn: os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise EnvironmentError("Missing GEMINI_MODEL")

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# 1) Node phân loại truy vấn
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Phân loại truy vấn khách hàng vào một trong các nhóm: Technical, Billing, General.\nQuery: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

# 2) Node phân tích cảm xúc
def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Phân tích cảm xúc của truy vấn khách hàng. Trả về Positive, Neutral hoặc Negative.\nQuery: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

# 3) Node tạo phản hồi

def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Cung cấp phản hồi hỗ trợ kỹ thuật cho truy vấn: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
    return {"response": chain.invoke({"query": state["query"]}).content}


def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Cung cấp phản hồi hỗ trợ thanh toán cho truy vấn: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
    return {"response": chain.invoke({"query": state["query"]}).content}


def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Cung cấp phản hồi chung cho truy vấn: {query}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
    return {"response": chain.invoke({"query": state["query"]}).content}

# 4) Node chuyển tiếp (escalation)

def escalate(state: State) -> State:
    return {"response": "Đã chuyển tiếp đến nhân viên hỗ trợ do cảm xúc tiêu cực."}

# 5) Logic điều hướng

def route_query(state: State) -> str:
    if state["sentiment"] == "Negative":
        return "escalate"
    return {
        "Technical": "handle_technical",
        "Billing":  "handle_billing",
    }.get(state["category"], "handle_general")

# --- Xây dựng StateGraph ---
workflow = StateGraph(State)
for name, fn in [
    ("categorize", categorize),
    ("analyze_sentiment", analyze_sentiment),
    ("handle_technical", handle_technical),
    ("handle_billing", handle_billing),
    ("handle_general", handle_general),
    ("escalate", escalate),
]:
    workflow.add_node(name, fn)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing":  "handle_billing",
        "handle_general":  "handle_general",
        "escalate":         "escalate",
    }
)
for end_node in ["handle_technical", "handle_billing", "handle_general", "escalate"]:
    workflow.add_edge(end_node, END)

workflow.set_entry_point("categorize")
app = workflow.compile()

# Hàm tiện ích để mở API

def run_customer_support(query: str) -> dict:
    res = app.invoke({"query": query})
    return {k: res[k] for k in ("category", "sentiment", "response")}
```

## 6. Xây dựng Server API (`backend/app.py`)

```python
# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from customer_support import run_customer_support

# Tải biến môi trường từ .env
load_dotenv()

class QueryIn(BaseModel):
    query: str

app = FastAPI(
    title="API Hỗ trợ LangGraph + Gemini",
    version="1.0"
)

@app.post("/support")
def support_endpoint(body: QueryIn):
    """POST /support -> {category, sentiment, response}"""
    return run_customer_support(body.query)

@app.get("/health")
def health_check():
    return {"status": "ok"}
```

## 7. Chạy trên máy cục bộ

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

* Truy cập [http://localhost:8000/docs](http://localhost:8000/docs) để xem giao diện tương tác của API.
* Thử nghiệm bằng `curl` hoặc Postman:

  ```bash
  curl -X POST http://localhost:8000/support \
       -H 'Content-Type: application/json' \
       -d '{"query":"Tôi không thể đăng nhập tài khoản!"}'
  ```

## 8. (Tùy chọn) Docker hóa

**Dockerfile**:

```dockerfile
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t support-agent .
docker run --rm -p 8000:8000 --env-file .env support-agent
```

## Kết luận

Bạn đã có một API hỗ trợ khách hàng chạy cục bộ, được xây dựng trên LangGraph và Google Gemini. Bạn có thể tiếp tục:

* Mở rộng logic của các node (ví dụ: lưu cache, ghi log, truy vấn cơ sở dữ liệu)
* Kết nối với pipeline webhook hoặc giao diện người dùng front-end
* Bảo mật bằng xác thực hoặc giới hạn tần suất yêu cầu

Chúc bạn xây dựng các workflow thông minh cùng LangGraph + Gemini!
