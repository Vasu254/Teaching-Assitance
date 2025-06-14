import os
import json
import re
import numpy as np
import tiktoken
from pathlib import Path
from tqdm import tqdm

# Import existing functions
from chunking import chunk_sentences_by_tokens
from embed import embed_texts, EMBEDDING_MODEL

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

# GitHub repository URL for source links
GITHUB_REPO_URL = "https://github.com/sanand0/tools-in-data-science-public/blob/main"


import re


def clean_markdown_content(content):
    """Clean markdown content while preserving URLs, images, code, and structure."""
    # Remove HTML comments
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    # Normalize excessive blank lines (keep double \n for paragraphs)
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

    lines = content.split("\n")
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Track code block state
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue

        if in_code_block:
            cleaned_lines.append(line)
            continue

        # Preserve lines that contain markdown links, images, or URLs
        if (
            re.search(r"\[.*?\]\(.*?\)", line)
            or "http" in line
            or "![" in line
            or stripped.startswith(("    ", "\t"))
        ):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(stripped)

    # Rejoin and collapse extra internal spaces (not after `:` or `/`)
    content = "\n".join(cleaned_lines)
    content = re.sub(r"(?<!:)(?<!/)  +", " ", content)

    return content.strip()


def extract_markdown_files():
    """Extract all markdown files and their content."""
    md_files = []
    repo_path = Path("tools-in-data-science-public")

    for md_file in repo_path.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            cleaned_content = clean_markdown_content(content)

            if cleaned_content:
                # Use filename without extension as ID
                file_id = md_file.stem
                # Use filename as title (can be improved later)
                title = md_file.stem.replace("-", " ").replace("_", " ").title()

                md_files.append(
                    {
                        "file_id": file_id,
                        "title": title,
                        "file_path": str(md_file.relative_to(repo_path)),
                        "content": cleaned_content,
                    }
                )

        except Exception:
            continue

    return md_files


def make_md_chunks(md_files):
    """Create chunks from markdown files using existing chunking function."""
    all_chunks = []

    for md_file in md_files:
        file_id = md_file["file_id"]
        title = md_file["title"]
        content = md_file["content"]
        relative_path = md_file["file_path"]  # e.g., "week1/intro.md"

        if not content.strip():
            continue

        # Create chunks from the content using existing function
        content_chunks = chunk_sentences_by_tokens(content)
        source_url = f"{GITHUB_REPO_URL}/{relative_path}"

        for i, chunk in enumerate(content_chunks, start=1):
            token_count = len(tokenizer.encode(chunk))
            if token_count > 500:  # MAX_TOKENS from chunking.py
                continue

            all_chunks.append(
                {
                    "file_id": file_id,
                    "chunk_id": f"{file_id}_{i}",
                    "title": title,
                    "url": source_url,
                    "text": chunk,
                    "source_type": "course_content",
                }
            )

    return all_chunks


def create_course_embeddings(chunks, batch_size=50):
    """Create embeddings for course content chunks using existing embed function."""
    embeddings_matrix = []
    metadata = []

    for i in tqdm(
        range(0, len(chunks), batch_size), desc="Embedding course content chunks"
    ):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(texts)

        for chunk, embedding in zip(batch, embeddings):
            embeddings_matrix.append(embedding)
            metadata.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file_id": chunk["file_id"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "text": chunk["text"],
                    "source_type": chunk["source_type"],
                }
            )

    return np.array(embeddings_matrix), np.array(metadata, dtype=object)


def combine_embeddings():
    """Combine existing embeddings with course content embeddings into main.npz."""
    # Load existing embeddings
    existing_data = np.load("embedded_chunks.npz", allow_pickle=True)
    existing_embeddings = existing_data["embeddings"]
    existing_metadata = existing_data["metadata"]

    # Load course content embeddings if they exist
    if os.path.exists("course_content.npz"):
        course_data = np.load("course_content.npz", allow_pickle=True)
        course_embeddings = course_data["embeddings"]
        course_metadata = course_data["metadata"]

        # Combine embeddings and metadata
        combined_embeddings = np.vstack([existing_embeddings, course_embeddings])
        combined_metadata = np.concatenate([existing_metadata, course_metadata])
    else:
        # If no course content, use existing only
        combined_embeddings = existing_embeddings
        combined_metadata = existing_metadata

    # Save combined embeddings
    np.savez_compressed(
        "main.npz", embeddings=combined_embeddings, metadata=combined_metadata
    )

    return len(combined_embeddings)


def main():
    md_files = extract_markdown_files()

    # Save cleaned markdown content
    with open("course_content.json", "w", encoding="utf-8") as f:
        json.dump(md_files, f, indent=2, ensure_ascii=False)

    chunks = make_md_chunks(md_files)

    # Save chunks
    with open("course_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    embeddings, metadata = create_course_embeddings(chunks)

    # Save NPZ format
    np.savez_compressed("course_content.npz", embeddings=embeddings, metadata=metadata)

    # Create combined embeddings file
    total_embeddings = combine_embeddings()

    return {
        "markdown_files": len(md_files),
        "chunks": len(chunks),
        "course_embeddings": len(embeddings),
        "total_embeddings": total_embeddings,
    }


if __name__ == "__main__":
    main()
