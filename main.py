# File: server.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from process import get_chunks, get_embeddings, question_answering

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

app = FastAPI()

# Define OpenAPI security schema to enable Swagger "Authorize" button
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="RAG API",
        version="1.0.0",
        description="RAG backend API with Bearer token authorization",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if method in ["post", "get"]:
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


class QueryRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: Request, payload: QueryRequest, authorization: Optional[str] = Header(None)):
    # --- Bearer Token Check ---
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Step 1: Chunk the document
        chunks = get_chunks(payload.documents)

        # Step 2: Embed the chunks
        embeddings = get_embeddings(chunks)

        # Step 3: Answer each question
        answers = []
        for question in payload.questions:
            answer = question_answering(question, embeddings, chunks)
            answers.append(answer)

        return {"answers": answers}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

