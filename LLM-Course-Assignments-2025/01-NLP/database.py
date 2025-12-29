import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os


class MedicalVectorDatabase:
    def __init__(self, model_name="moka-ai/m3e-base"):
        """
        初始化医疗向量数据库
        """
        self.embed_model = SentenceTransformer(model_name)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.metadata = []

    def build_from_chunks(self, chunks_file="medical_chunks.json", batch_size=1000):
        """
        从chunks文件构建向量数据库
        """
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeds = self.embed_model.encode(
                batch_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeds)

        embeddings = np.vstack(embeddings)
        self.index.add(embeddings)
        self.chunks = chunks
        self.metadata = [chunk.get("metadata", {}) for chunk in chunks]

    def search(self, query, top_k=5, threshold=0.5):
        """
        检索相似文档
        """
        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue

            if score < threshold:
                continue

            result = {
                "id": int(idx),
                "chunk": self.chunks[idx],
                "score": float(score),
                "text": self.chunks[idx]["text"],
                "metadata": self.metadata[idx]
            }
            results.append(result)

        return results

    def save(self, path="medical_vector_db"):
        """保存向量数据库"""
        faiss.write_index(self.index, f"{path}.index")

        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f)

    def load(self, path="medical_vector_db"):
        """加载向量数据库"""
        self.index = faiss.read_index(f"{path}.index")

        with open(f"{path}.meta.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]


def build_vector_database():
    """
    构建向量数据库
    """
    vector_db = MedicalVectorDatabase(model_name="moka-ai/m3e-base")

    if os.path.exists("medical_chunks.json"):
        vector_db.build_from_chunks()
        vector_db.save()
        return vector_db
    else:
        print("未找到medical_chunks.json文件")
        return None


def load_vector_database():
    """
    加载已存在的向量数据库
    """
    if os.path.exists("medical_vector_db.index") and os.path.exists("medical_vector_db.meta.pkl"):
        vector_db = MedicalVectorDatabase(model_name="moka-ai/m3e-base")
        vector_db.load()
        return vector_db
    return None


def get_vector_database():
    """
    获取向量数据库实例，如果不存在则创建
    """
    vector_db = load_vector_database()
    if vector_db is None:
        vector_db = build_vector_database()
    return vector_db


def query_vector_database(vector_db, query, top_k=5):
    """
    查询向量数据库
    """
    if vector_db is None:
        print("向量数据库未初始化")
        return []

    results = vector_db.search(query, top_k)
    return results


if __name__ == "__main__":
    # 获取向量数据库
    vector_db = get_vector_database()

    if vector_db:
        # 测试检索
        test_queries = [
            "高血压应该吃什么药？",
            "感冒了怎么办？",
            "胃痛怎么缓解？",
            "糖尿病患者可以吃什么水果？"
        ]

        for query in test_queries:
            results = query_vector_database(vector_db, query, top_k=3)

            if results:
                print(f"查询: {query}")
                for i, result in enumerate(results):
                    print(f"  结果 {i + 1} (相似度: {result['score']:.3f}):")
                    print(f"    科室: {result['metadata']['department']}")
                    print(f"    内容: {result['text'][:80]}...")
                print()