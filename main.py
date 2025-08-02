# File: server.py
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from process import get_chunks, get_embeddings, question_answering

app = FastAPI()

# Replace with your actual token in production
auth_token = "93f200ce0d5548d7c542c6887518de76c661cba038ed34769b2569aa72b3932d"

class QueryRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: Request, payload: QueryRequest, authorization: Optional[str] = Header(None)):
    # --- Bearer Token Check ---
    if authorization != f"Bearer {auth_token}":
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
