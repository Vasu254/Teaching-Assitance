from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import json
import numpy as np
from google import genai
from google.genai import types
import mimetypes
import base64
from openai import OpenAI
from dotenv import load_dotenv
from collections import deque
import time
from typing import Optional
import uvicorn
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None


# Initialize clients
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url="https://aipipe.org/openai/v1"
)
request_timestamps = deque()


def load_embeddings():
    """Load embeddings and metadata from NPZ file."""
    data = np.load("../data_source/main.npz", allow_pickle=True)
    return data["embeddings"], data["metadata"]


def get_embedding(text):
    """Get embedding for query text."""
    response = openai_client.embeddings.create(
        input=[text], model="text-embedding-3-small", encoding_format="float"
    )
    return np.array(response.data[0].embedding)


def process_image_with_gemini(image_data, mime_type):
    """Process image with Gemini API."""
    global request_timestamps

    # Rate limiting
    while len(request_timestamps) >= 15:
        elapsed = time.time() - request_timestamps[0]
        if elapsed < 60:
            time.sleep(60 - elapsed)
        else:
            request_timestamps.popleft()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    request_timestamps.append(time.time())

    image = types.Part.from_bytes(data=image_data, mime_type=mime_type)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "Extract the content from this image, preserving its structure, formatting, and context. Do not add any introductions or explanations — output only the image's content.",
            image,
        ],
    )
    return response.text.strip()


@app.post("/api/")
async def ask(request: QueryRequest):
    try:
        embeddings, metadata = load_embeddings()

        query_text = request.question
        image_text = ""

        # Process image if provided
        if request.image:
            try:
                # Decode base64 image
                image_data = base64.b64decode(request.image)
                # Try to determine mime type from image data
                mime_type = "image/jpeg"  # Default
                if image_data.startswith(b"\x89PNG"):
                    mime_type = "image/png"
                elif image_data.startswith(b"\xff\xd8"):
                    mime_type = "image/jpeg"
                elif image_data.startswith(b"GIF"):
                    mime_type = "image/gif"
                elif image_data.startswith(b"RIFF") and b"WEBP" in image_data[:12]:
                    mime_type = "image/webp"

                image_content = process_image_with_gemini(image_data, mime_type)
                image_text = f" {image_content}"
            except Exception as e:
                print(f"Error processing image: {e}")

        # Combine query and image text
        full_query = f"{query_text} {image_text}".strip()
        # Get query embedding and find similar documents
        query_embedding = get_embedding(full_query)
        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-10:][::-1]

        # Get top documents with token limit
        context_docs = []
        total_tokens = 0
        for idx in top_indices:
            doc = metadata[idx]
            doc_tokens = len(doc["text"].split()) * 1.3  # Rough token estimate
            if total_tokens + doc_tokens > 500:
                break
            context_docs.append(doc)
            total_tokens += doc_tokens

        # Generate response with Gemini
        context = "\n\n".join(
            [
                f"Title: {doc['title']}\nURL: {doc['url']}\nContent: {doc['text']}"
                for doc in context_docs
            ]
        )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = f"Context:\n{context}\n\nQuestion: {full_query}\n\n You are a knowledgeable virtual teaching assistant for the Tools for Data Science (TDS) course."
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt]
        )

        # Format response
        result = {
            "answer": response.text.strip(),
            "links": [
                {"url": doc["url"], "text": doc["title"]} for doc in context_docs
            ],
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
