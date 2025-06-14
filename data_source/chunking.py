import json
import re
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 500
OVERLAP = 30


def split_by_token_limit(text, token_limit=MAX_TOKENS, overlap=OVERLAP):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + token_limit, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())

        start += token_limit - overlap

    return chunks


def chunk_sentences_by_tokens(text, token_limit=MAX_TOKENS, overlap=OVERLAP):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = tokenizer.encode(sent)

        if len(sent_tokens) > token_limit:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            long_chunks = split_by_token_limit(sent, token_limit, overlap)
            chunks.extend(long_chunks)
            continue

        if current_tokens + len(sent_tokens) > token_limit:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            overlap_tokens = tokenizer.encode(chunk_text)[-overlap:]
            overlap_text = tokenizer.decode(overlap_tokens)
            current = [overlap_text]
            current_tokens = len(overlap_tokens)

        current.append(sent)
        current_tokens += len(sent_tokens)

    if current:
        chunks.append(" ".join(current))

    return chunks


def make_chunks(topics):
    all_chunks = []
    for topic in topics:
        topic_id = topic["topic_id"]
        title = topic.get("title", "")
        posts = topic.get("posts", [])

        for post_no, post in enumerate(posts, start=1):
            username = post.get("username", "")
            text = post.get("text", "").strip()

            if not text:
                continue

            post_text = f"{username}:\n{text}"
            post_chunks = chunk_sentences_by_tokens(post_text)

            for i, chunk in enumerate(post_chunks, start=1):
                token_count = len(tokenizer.encode(chunk))
                if token_count > MAX_TOKENS:
                    print(
                        f"⚠️ Chunk too long in topic {topic_id}, post {post_no}, chunk {i}: {token_count} tokens"
                    )

                all_chunks.append(
                    {
                        "topic_id": topic_id,
                        "title": title,
                        "post_id": f"{post_no}",
                        "chunk_id": f"{topic_id}_{post_no}_{i}",
                        "text": chunk,
                    }
                )

    return all_chunks


if __name__ == "__main__":
    with open("tds_cleaned_data.json", "r") as f:
        topics = json.load(f)

    chunks = make_chunks(topics)

    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Chunking complete: {len(chunks)} chunks written to chunks.json")
