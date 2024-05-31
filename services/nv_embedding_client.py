import requests
from typing import List, Dict
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

class NVEmbeddings(Embeddings):
    def __init__(self, api_url, model):
        self.api_url = api_url
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.body = {
            "model": model,
        }
    
    def embed_documents(self, texts: List[str]) -> List[Document]:
        self.body.update({
            "input": texts,
            "input_type": "passage"
        })

        response = requests.post(self.api_url, headers=self.headers, json=self.body)

        if response.status_code != 200:
            raise ValueError(f"Request failed with status {response.status_code}")
        
        data = response.json()['data']
        
        return [passage_embedding['embedding'] for passage_embedding in data]
    
    def embed_query(self, query:str) -> List[str]:
        self.body.update({
            "input": [query],
            "input_type": "query"
        })

        response = requests.post(self.api_url, headers=self.headers, json=self.body)

        if response.status_code != 200:
            raise ValueError(f"Request failed with status {response.status_code}")
        
        return response.json()['data'][0]['embedding']