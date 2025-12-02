import logging
import os
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="RAG_App",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)

@inngest_client.create_function(
    fn_id="RAG: Ingest",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> dict:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id).model_dump()

    def _upsert(chunks_and_src_dict: dict) -> dict:
        chunks_and_src = RAGChunkAndSrc(**chunks_and_src_dict)
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source_id": source_id, "text": chunks[i]} for i in range(len(chunks))]
        with QdrantStorage() as store:
            store.upsert_vectors(ids, vecs, payloads)
        return RAGUpsertResult(inngested=len(chunks)).model_dump()

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx))
    inngested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src))
    return inngested

@inngest_client.create_function(
    fn_id="RAG: QueryPDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> dict:
        query_vec = embed_texts([question])[0]
        with QdrantStorage() as store:
            raw_found = store.search_vectors(query_vec, top_k=500)
        
        results = raw_found.get("results", [])
        
        terms = {t for t in question.lower().split() if len(t) >= 3}
        
        for r in results:
            source_lower = r["source"].lower()
            matches = sum(1 for term in terms if term in source_lower)
            if matches > 0:
                r["score"] += 0.5 + (matches * 0.1)
                        
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        contexts = [r["text"] for r in top_results]
        sources = list(set(r["source"] for r in top_results))
        
        return RAGSearchResult(contexts=contexts, sources=sources).model_dump()

    question = ctx.event.data.get("question")
    if not question:
        raise ValueError("Missing 'question' in event data. Please provide a question to query.")
    top_k = int(ctx.event.data.get("top_k", 15))

    found_dict = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))
    found = RAGSearchResult(**found_dict)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "use the following context to answer the question.\n\n"
        f"context:\n{context_block}\n\n"
        f"question: {question}\n\n"
        "Answer consisely using the context above"
    )
    
    # Using api_key assuming it's the standard for the adapter
    adapter = ai.openai.Adapter(
        base_url="https://openrouter.ai/api/v1",
        auth_key=os.getenv("OPENROUTER_API_KEY"),
        model="x-ai/grok-4.1-fast:free"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 100,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_inngest_pdf, rag_query_pdf_ai])