import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Configure OpenAI client with aipipe proxy
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url="https://aipipe.org/openai/v1"
)

EMBEDDING_MODEL = "text-embedding-3-small"
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/t"


def get_post_url(topic_id: int, post_id: str) -> str:
    return f"{DISCOURSE_BASE_URL}/{topic_id}/{post_id}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of texts."""
    response = client.embeddings.create(
        input=texts, model=EMBEDDING_MODEL, encoding_format="float"
    )
    return [item.embedding for item in response.data]


def load_embeddings_npz(npz_file="embedded_chunks.npz"):
    """Load embeddings and metadata from compressed NPZ file."""
    data = np.load(npz_file, allow_pickle=True)
    return data["embeddings"], data["metadata"]


def prepare_embeddings(
    chunks_file="chunks.json", output_file="embedded_chunks.jsonl", batch_size=50
):
    with open(chunks_file, "r") as f:
        chunks = json.load(f)

    embedded_data = []
    embeddings_matrix = []
    metadata = []

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_texts(texts)

        for chunk, embedding in zip(batch, embeddings):
            record = {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "text": chunk["text"],
                "embedding": embedding,
                "url": get_post_url(chunk["topic_id"], chunk["post_id"]),
            }
            embedded_data.append(record)

            # For NPZ format
            embeddings_matrix.append(embedding)
            metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "url": get_post_url(chunk["topic_id"], chunk["post_id"]),
                }
            )

    # Save JSONL format
    with open(output_file, "w") as f:
        for record in embedded_data:
            f.write(json.dumps(record) + "\n")

    # Save NPZ format
    npz_file = output_file.replace(".jsonl", ".npz")
    np.savez_compressed(
        npz_file,
        embeddings=np.array(embeddings_matrix),
        metadata=np.array(metadata, dtype=object),
    )

    print(
        f"✅ Embedding complete: {len(embedded_data)} entries written to {output_file}"
    )
    print(f"✅ Compressed embeddings saved to {npz_file}")


if __name__ == "__main__":
    prepare_embeddings()
