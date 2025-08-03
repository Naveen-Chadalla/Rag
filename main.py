# File: server.py
import os
import time
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from process import get_chunks, get_embeddings, question_answering

load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

app = FastAPI()



class QueryRequest(BaseModel):
    documents: str  # URL to the PDF document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: Request, payload: QueryRequest, authorization: Optional[str] = Header(None)):
    overall_start = time.time()
    # --- Bearer Token Check ---
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Step 1: Chunk the document
        start = time.time()
        chunks = get_chunks(payload.documents)
        print(f"✅ Chunking completed in {time.time() - start:.2f} seconds")
        
        # Step 2: Embed the chunks
        start = time.time()
        embeddings = get_embeddings(chunks)
        print(f"✅ Embedding completed in {time.time() - start:.2f} seconds")
        
        # Step 3: Answer each question
        answers = []
        idx = 0
        for question in payload.questions:
            start = time.time()
            print(f"🔹 Answering question {idx+1}: {question[:15]}...")
            answer = question_answering(question, embeddings, chunks)
            print(f"✅ Answered Q{idx} in {time.time() - start:.2f} seconds")
            answers.append(answer)
        total_time = time.time() - overall_start
        print(f"🎯 Total request time: {total_time:.2f} seconds")
        print("Returning answers:", answers)

        return {"answers": answers}
    except Exception as e:
        print(traceback.format_exc()) 
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})






