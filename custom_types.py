import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str | None = None

class RAGUpsertResult(pydantic.BaseModel):
    inngested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    num_contexts: int
    sources: list[str]
