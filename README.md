# RAG PDF Assistant with 3D Visualization

A robust Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs, ask questions about their content, and interactively visualize the semantic vector space in 3D.

## Features

-   **PDF Ingestion**: robustly parses and chunks PDF documents using `LlamaIndex`.
-   **Durable Workflows**: Uses **Inngest** to manage background ingestion jobs, ensuring reliability and retries.
-   **Vector Search**: Stores embeddings in **Qdrant** (local mode) for fast similarity search.
-   **Interactive AI**: Queries are processed by an LLM (via OpenRouter/OpenAI) to provide context-aware answers.
-   **3D Visualization**: Visualizes document chunks and user queries in a 3D semantic space using **PCA** and **Plotly**.
-   **Source Management**: Easily manage and delete uploaded documents directly from the UI.

## Tech Stack

-   **Frontend**: Streamlit
-   **Backend**: FastAPI
-   **Orchestration**: Inngest
-   **Vector DB**: Qdrant
-   **AI/ML**: LlamaIndex, OpenAI (Embeddings), Scikit-learn (PCA)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up the environment**:
    Create a `.env` file in the root directory:
    ```env
    OPENROUTER_API_KEY=your_api_key_here
    INNGEST_API_BASE=http://127.0.0.1:8288/v1
    ```

3.  **Install dependencies**:
    Using `uv` (recommended) or pip:
    ```bash
    uv sync
    # OR
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the Inngest Dev Server** (in a separate terminal):
    ```bash
    npx inngest-cli@latest dev
    ```

2.  **Start the FastAPI Backend**:
    ```bash
    uv run uvicorn main:app --reload
    ```

3.  **Start the Streamlit Frontend**:
    ```bash
    uv run streamlit run streamlit_app.py
    ```

4.  **Open your browser**:
    -   Streamlit App: `http://localhost:8501`
    -   Inngest Dashboard: `http://localhost:8288`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
