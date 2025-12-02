import pandas as pd
from sklearn.decomposition import PCA
from vector_db import QdrantStorage
import plotly.express as px

def get_visualization_data(query_vector=None, query_text=None):
    """
    Fetches all vectors from Qdrant, reduces dimensions to 3D using PCA,
    and returns a DataFrame and a Plotly figure.
    """
    with QdrantStorage() as store:
        # 1. Fetch all vectors
        # Note: For very large datasets, this might need pagination/optimization.
        # Current implementation assumes a manageable size for a demo.
        points = []
        next_offset = None
        
        while True:
            batch, next_offset = store.client.scroll(
                collection_name=store.collection,
                limit=100,
                with_payload=True,
                with_vectors=True,
                offset=next_offset
            )
            points.extend(batch)
            if next_offset is None:
                break
            
    if not points:
        return None, None

    # 2. Prepare data structure
    vectors = []
    metadata = []
    
    for p in points:
        vectors.append(p.vector)
        payload = p.payload or {}
        text_preview = payload.get("text", "")[:100] + "..." if payload.get("text") else "No text"
        metadata.append({
            "source": payload.get("source_id", "Unknown"),
            "text": text_preview,
            "type": "Document",
            "size": 5  # Base size for documents
        })
        
    # 3. Handle Query Vector
    if query_vector is not None:
        vectors.append(query_vector)
        metadata.append({
            "source": "User Query",
            "text": query_text or "Query",
            "type": "Query",
            "size": 12  # Significantly larger for query
        })

    # 4. Dimensionality Reduction (PCA)
    # We need at least 3 samples to do 3-component PCA effectively, 
    # but sklearn handles n_samples < n_components gracefully (components will be less).
    n_components = 3
    if len(vectors) < 3:
        # Fallback for very few points? Or just let PCA handle it (it produces min(n_samples, n_components))
        pass

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(vectors)
    
    # Pad with zeros if we have fewer than 3 dimensions (e.g. only 1 or 2 points)
    # This ensures the DataFrame always has x, y, z columns.
    if embeddings_3d.shape[1] < 3:
        import numpy as np
        padding = np.zeros((embeddings_3d.shape[0], 3 - embeddings_3d.shape[1]))
        embeddings_3d = np.hstack((embeddings_3d, padding))

    df = pd.DataFrame(metadata)
    df["x"] = embeddings_3d[:, 0]
    df["y"] = embeddings_3d[:, 1]
    df["z"] = embeddings_3d[:, 2]
    
    # 5. Create Plot
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="source",
        symbol="type",
        size="size",  # Use dynamic sizing
        hover_data=["text"],
        opacity=0.8,
        size_max=15,  # Allow larger marker variation
        height=800,   # Make the plot taller
        title="Semantic Space Visualization"
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            aspectmode='cube',  # Forces a cubic aspect ratio (not flattened)
            xaxis=dict(title='', showticklabels=False),
            yaxis=dict(title='', showticklabels=False),
            zaxis=dict(title='', showticklabels=False),
        ),
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    return df, fig
