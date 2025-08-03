#import the necessaries
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load the env fie
load_dotenv()

#Access the API key and the embeding model from the env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Load the model
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Import and initialize the Gemini AI client from Google's generative AI library.
# This client will be used to interact with the Gemini API for generating text.

# Replace 'GOOGLE_API_KEY' with your actual API key.
client = genai.Client(api_key= GOOGLE_API_KEY)


def get_chunks(url):
    """
    Downloads a PDF from the given URL, extracts its text, splits it into manageable chunks,
    and returns those chunks.

    Args:
        url (str): The direct URL to a PDF file.

    Returns:
        list[str]: A list of text chunks extracted and split from the PDF content.
    """
    
    #Send an HTTP GET request to fetch the PDF file from the given URL
    response = requests.get(url)

    #Read the binary content of the response into a BytesIO buffer
    # This allows the PDF to be read in-memory without saving it to disk
    pdf_file = BytesIO(response.content)

    #Initialize the PDF reader to parse the PDF content
    reader = PdfReader(pdf_file)

    #Initialize an empty string to store all extracted text from the PDF
    text = ""

    #Loop through each page in the PDF and extract the text
    # Append each page's text with a newline separator
    for page in reader.pages:
        text += page.extract_text() + '\n'

    #Initialize RecursiveCharacterTextSplitter to split the extracted text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", ""],
    )

    #Split the large text into smaller chunks using the configured splitter
    chunks = text_splitter.split_text(text)

    #Return the splitted chunks.
    return chunks



def get_embeddings(chunks):
    """
    Converts a list of text chunks into numerical vector embeddings using a preloaded embedding model.

    Args:
        chunks (list[str]): A list of text strings (typically segments of a larger document).

    Returns:
        torch.Tensor: A tensor containing embeddings for each input text chunk.
    """

    # Convert each chunk into an embedding
    embeddings = embedder.encode(chunks,convert_to_tensor=True)

    # Return the embeddings
    return embeddings


def question_answering(question, embeddings, chunks):
    """
    Retrieves the answer for the question based on the provided embeddings and chunks

    Args:
        chunks (list[str]): A list of text strings (typically segments of a larger document).

        question (str): The user's input query, typically a natural language question or sentence.

        embeddings (torch.Tensor): A set of vector embeddings representing chunks of text or documents.
            These should be the same type as returned by your embedding model

    Returns:
        Answer: answer to the question provided.
    """
    # get the  embeddigs of the question and convert them to tensor object 
    question_embedding = embedder.encode(question, convert_to_tensor=True)

    # Define top k elements that need to be retrieved 
    top_k = 5

    # Do Semantic Search on the question embedding and all embeddings and retrieve top k results 
    # It gives indexes of top k embeddings
    hits = util.semantic_search(question_embedding, embeddings, top_k=top_k)[0]

    # Define a retrieved_chunks to store the retrieved chunks
    retrieved_chunks = ""

    # Loop through each result in 'hits', which is typically a list of ranked search results.
    # Each 'hit' is expected to be a dictionary with metadata about the retrieved embedding indexes
    for hit in hits:

        # Extract the index of the matching chunk from the search result
        idx = hit['corpus_id']

        # Retrieve the matched chunk and the next chunk (idx+1) from the list of original text chunks.
        # This is done to provide additional context following the matched chunk.
        retrieved_chunks = retrieved_chunks + chunks[idx] + chunks[idx+1]
    
    # Construct a formatted multi-line string that serves as a prompt for a language model or answer generation task.
    # The prompt includes:
        # - The user's original question (`question`)
        # - Relevant contextual information pulled from the document chunks (`retrieved_chunks`)
    content = f'''
    USER QUERY : {question}

    CONTEXT : {retrieved_chunks}

    ANSWER :
    '''

    # This is the system prompt that is given to the llm
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

    # Send the prepared prompt to the Gemini model for content generation.
    # This call uses the Gemini API client to generate a response based on the given content.
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",# Specify the model version to use (lightweight and fast version of Gemini 2.5)

        # Configuration settings for content generation
        config=types.GenerateContentConfig(
            system_instruction = instruction,   # Optional system-level instruction to guide the model's behavior
            temperature = 2  # Controls the randomness of the output (higher = more creative, lower = more deterministic)
            ),
        contents = content  # The actual prompt, including the user query and context
    )
    # Return the text response
    return response.text


