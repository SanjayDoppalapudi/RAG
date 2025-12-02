from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

class QdrantStorage:
    def __init__(self, collection="docs_2560", dim=2560):
        self.client = QdrantClient(path="qdrant_storage", prefer_grpc=False, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def recreate_collection(self):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    def upsert_vectors(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search_vectors(self, query_vector, top_k: int = 5, source_id: str = None, score_threshold: float = 0.1):
        query_filter = None
        if source_id:
            query_filter = Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            )
            
        points = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold
        ).points
        
        results = []
        for p in points:
            payload = p.payload or {}
            text = payload.get("text")
            if text:
                results.append({
                    "text": text,
                    "source": payload.get("source_id", ""),
                    "score": p.score
                })

        return {
            "context": [r["text"] for r in results],
            "sources": list({r["source"] for r in results}),
            "results": results
        }

    def delete_source(self, source_id: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(value=source_id)
                    )
                ]
            )
        )

    def list_sources(self):
        sources = set()
        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_offset
            )
            for p in batch:
                payload = p.payload or {}
                s = payload.get("source_id")
                if s:
                    sources.add(s)
            if next_offset is None:
                break
        return sorted(list(sources))

    def close(self):
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    