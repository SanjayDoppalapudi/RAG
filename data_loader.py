from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
EMBED_MODEL = "qwen/qwen3-embedding-4b"
EMBED_DIM = 2560

splitter = SentenceSplitter(chunk_size = 1000, chunk_overlap = 200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, 'text', None)]
    if not texts or all(not t.strip() for t in texts):
        raise ValueError(f"No text extracted from {path}. The PDF might be a scanned image or empty.")
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]