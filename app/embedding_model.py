# app/embedding_model.py
import spacy

class EmbeddingModel:
    def __init__(self):
        
        self.nlp = spacy.load("en_core_web_md")

    def get_embedding(self, word: str) -> list[float]:
        
        # 让 spacy 处理这个词
        token = self.nlp(word)
        # 提取它的向量，并转换成 Python 的列表格式返回
        return token.vector.tolist()
