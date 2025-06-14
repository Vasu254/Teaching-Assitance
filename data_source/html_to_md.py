from collections import deque
from bs4 import BeautifulSoup
import json
import base64
import mimetypes
import requests
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()

DESC_FILE = "image_descriptions.json"

# -- Load existing descriptions once --
if os.path.exists(DESC_FILE):
    with open(DESC_FILE, "r", encoding="utf-8") as f:
        try:
            existing_descriptions = json.load(f)
        except json.JSONDecodeError:
            existing_descriptions = []
else:
    existing_descriptions = []

# Helper for lookup
description_lookup = {
    item["src"]: item["description"] for item in existing_descriptions
}


# Keep track of timestamps of requests
request_timestamps = deque()


def describe_image(src):
    global request_timestamps

    try:
        # Rate limit: 15 requests per 60 seconds
        while len(request_timestamps) >= 15:
            elapsed = time.time() - request_timestamps[0]
            if elapsed < 60:
                sleep_time = 60 - elapsed
                print(f"Rate limit hit. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                request_timestamps.popleft()

        response = requests.get(src, timeout=5)
        response.raise_for_status()
        image_bytes = response.content
        mime_type = mimetypes.guess_type(src)[0] or "image/jpeg"

        image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        # Load API key and make request
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        # Register current timestamp before calling
        request_timestamps.append(time.time())

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Extract the content from this image, preserving its structure, formatting, and context. Do not add any introductions or explanations — output only the image's content.",
                image,
            ],
        )

        return response.text.strip()

    except Exception as e:
        print(f"[describe_image ERROR] {e}")
        return f"[Error describing image: {str(e)}]"


def is_real_image(src, alt, title):
    if not src:
        return False
    if "europe1.discourse-cdn.com" in src and "/uploads/" in src:
        return True
    return False


def replace_images_with_description(soup):
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")

        # Remove base64 images directly
        if src and src.startswith("data:image"):
            img.decompose()
            continue

        if is_real_image(src, img.get("alt", ""), img.get("title", "")):
            if src in description_lookup:
                description = description_lookup[src]
            else:
                description = describe_image(src)
                description_lookup[src] = description
                existing_descriptions.append({"src": src, "description": description})
                with open(DESC_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing_descriptions, f, indent=2, ensure_ascii=False)

            description_text = f"\nIMAGE CONTENT: {description}\n"
            try:
                img.replace_with(description_text)
            except Exception as e:
                print(f"[replace_with ERROR] {e}")
                img.decompose()
        else:
            img.decompose()
    return soup


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    soup = replace_images_with_description(soup)
    return soup.get_text(separator="\n").strip()


# -- Main logic --
filepath = "tds_raw_html.json"
with open(filepath, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

for topic in raw_data:
    print(f"Processing Topic ID: {topic['topic_id']}, Title: {topic['title']}")
    for post in topic["posts"]:
        html_content = post.get("raw_html", "")
        if html_content:
            try:
                text = extract_text(html_content)
            except Exception as e:
                text = f"[Error processing HTML: {e}]"
            post["text"] = text
        else:
            post["text"] = ""
        post.pop("raw_html", None)

# Save cleaned output
newfilepath = "tds_cleaned_data.json"
with open(newfilepath, "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)

print(f"Processed data saved to {newfilepath}")
