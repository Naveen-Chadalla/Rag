# processor.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Load the model
embedder = SentenceTransformer(EMBEDDING_MODEL)

client = genai.Client(api_key= GOOGLE_API_KEY)


def get_chunks(url):
    
    # Download the PDF
    response = requests.get(url)
    pdf_file = BytesIO(response.content)

    # Read the PDF
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        # Extract the text from each page
        text += page.extract_text() + '\n'

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", ""],
    )
    chunks = text_splitter.split_text(text)

    return chunks



def get_embeddings(chunks):

    # Convert each chunk into an embedding
    embeddings = embedder.encode(chunks,convert_to_tensor=True)

    return embeddings

def question_answering(question, embeddings, chunks):

    question_embedding = embedder.encode(question, convert_to_tensor=True)

    top_k = 5

    hits = util.semantic_search(question_embedding, embeddings, top_k=top_k)[0]


    retrieved_chunks = ""

    for hit in hits:
        idx = hit['corpus_id']
        retrieved_chunks = retrieved_chunks + chunks[idx] + chunks[idx+1]
        # print(f"> Chunk: {idx} | Chunk: {chunks[idx]}")
    
    content = f'''
    USER QUERY : {question}

    CONTEXT : {retrieved_chunks}

    ANSWER : 
    '''
    instruction='''You are an intelligent assistant designed to answer user questions based only on the provided context and who talks like a human.

    Answer the question accurately and concisely using the information retrieved. 
    Do not make assumptions or fabricate answers.

    Say what you want to say in maximum three sentences.
    if it can be answered in one sentence then do so.
    Use simple and clear language.

    If the answer cannot be found in the given context, respond with:
    "I'm sorry, I couldn't find the relevant information in the provided documents."

    EXAMPLE QUESTION AND ANSWERS:

    question : What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
    answer : A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

    question : What is the waiting period for pre-existing diseases (PED) to be covered?
    answer : There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

    question : Does this policy cover maternity expenses, and what are the conditions?
    answer : Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.
    '''

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction = instruction,
            temperature = 2
            ),
        contents = content
    )

    return response.text
