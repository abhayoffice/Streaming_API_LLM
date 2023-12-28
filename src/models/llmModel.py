# my_llm.py

import asyncio
from typing import Any
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from langchain.agents import AgentType, initialize_agent
from src.db.faiss_db import create_faiss_db, load_faiss_db
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory


def my_llm_model(documents_db, embeddings):
    config = {'max_new_tokens': 1024,
              'repetition_penalty': 1.1,
              'temperature': 0.1,
              'top_k': 50,
              'top_p': 0.9,
              'context_length': 2048,
              'stream': True,
              'threads': int((os.cpu_count()))
              }

    llm = CTransformers(model='TheBloke/zephyr-7B-alpha-GGUF', model_file='zephyr-7b-alpha.Q5_K_S.gguf',
                        model_type="mistral", lib="avx2", config=config,
                        # callbacks=[StreamingStdOutCallbackHandler()],
                        callbacks=[],
                        streaming = True)

    documents_db = load_faiss_db(embeddings)
    retriever = documents_db.as_retriever(search_kwargs={'k': 2})

    template = ''' Strictly use the given context and answer the user's question. If the question is not related to the context simply say "The context is of unknown origin"
    Context={context}
    Question={question}
    Answer:
    '''
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        chain_type="stuff",
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    return chain

