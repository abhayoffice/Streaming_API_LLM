import asyncio
import os
import logging
from datetime import timedelta

from fastapi import FastAPI, HTTPException, status, Depends, Query
from fastapi.security import OAuth2PasswordRequestForm
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from starlette.middleware.cors import CORSMiddleware
from langchain.document_loaders import DirectoryLoader
from starlette.responses import StreamingResponse
from src.db.faiss_db import create_faiss_db
from src.models import llmModel
from src.security.auth import Token, authenticate_user, db, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, User, \
    get_current_active_user

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Determine the current directory of the script
# src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# Set the paths for reader and templates using the current directory
data_dir = os.path.join(src_dir, "data")  # Replace "data" with the relative path to your data directory


#Load the documents.
loader=DirectoryLoader(data_dir,show_progress=True)
documents=loader.load()

# Load the documents.
try:
    # Initialize Faiss database
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={'device': 'cpu'},
    )

    loader = DirectoryLoader(data_dir, show_progress=True)
    documents = loader.load()
    document_db = create_faiss_db(documents, embeddings)
except Exception as e:
    logger.error(f"Error loading documents: {str(e)}")
    raise SystemExit("Exiting due to error in loading documents.")


# global chain  # Declare chain as a global variable

# Load the model
try:
    chain = llmModel.my_llm_model(document_db, embeddings)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise SystemExit("Exiting due to error in loading the model.")

app = FastAPI()

def result_with_sources(response):

  source_list=[]
  for source_item in response['source_documents']:
    source=source_item.metadata['source']

    file_name=os.path.basename(source)

    if file_name not in source_list:
      source_list.append(file_name)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################################################


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# @app.post("/chat")
# async def chat(query: Query, response: StreamingResponse):
#     async def stream():
#         async for token in callback_handler:
#             yield token
#
#     streaming_iterator = stream()
#     task = asyncio.create_task(agent(query, streaming_iterator))
#     return StreamingResponse(streaming_iterator)


####################################################################
@app.post('/query')
async def get_query(query_str: str, current_user: User = Depends(get_current_active_user)):
    responses = chain(query_str)

    async def generate():
        # Process your response and yield chunks for streaming
        # For example, assuming `response` is a string
        for chunk in responses.get('result'):
            yield chunk.encode("utf-8")
            print(chunk)

    return StreamingResponse(content=generate(), media_type="application/octet-stream")
    # output = {'response': responses.get('result')}
    # return output

#############################################
# @app.post('/query')
# async def get_query(query_str: str):
#     responses = dict(chain(query_str))
#     output = {'response': responses.get('result')}
#     return output

#Run the application from ehre.

if __name__ == "__main__":
    import uvicorn

    ssl_key_filename = "llm-devop_key.pem"
    ssl_cert_filename = "llm-devop_cert.pem"

    # Get the absolute paths to the SSL key and certificate files
    ssl_key_path = os.path.abspath(os.path.join(src_dir, "ssl_certi", ssl_key_filename))
    ssl_cert_path = os.path.abspath(os.path.join(src_dir, "ssl_certi", ssl_cert_filename))

    try:
        uvicorn.run(app, host="localhost", port=8000, ssl_keyfile=ssl_key_path, ssl_certfile=ssl_cert_path)
    except Exception as e:
        logger.error(f"Error running Uvicorn: {str(e)}")
        raise SystemExit("Exiting due to error in running Uvicorn.")