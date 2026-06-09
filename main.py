# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from app.embedding_model import EmbeddingModel

app = FastAPI()

spacy_model = EmbeddingModel()


class WordRequest(BaseModel):
    query_word: str

@app.get("/")
def read_root():
    return {"Message": "Welcome to the Spacy Embedding API!"}


@app.post("/get_embedding")
def fetch_vector(request: WordRequest):
   
    vector = spacy_model.get_embedding(request.query_word)
    
  
    return {
        "word": request.query_word,
        "embedding_length": len(vector), 
        "embedding_vector": vector       
    }
