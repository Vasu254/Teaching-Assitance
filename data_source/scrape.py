import requests
import json
import time
from datetime import datetime

# ---- CONFIG ----
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_SLUG = "courses/tds-kb"
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14)


# ---- LOAD COOKIES ----
def load_cookies(json_path):
    with open(json_path, "r") as f:
        cookies = json.load(f)
    return {
        c["name"]: c["value"]
        for c in cookies
        if "discourse.onlinedegree.iitm.ac.in" in c.get("domain", "")
    }


cookies = load_cookies("cookie.json")


# ---- GET TOPICS WITHIN DATE RANGE ----
def get_filtered_topics():
    page = 0
    filtered = []

    while True:
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}/l/latest.json?page={page}"
        res = requests.get(url, cookies=cookies)
        if res.status_code != 200:
            break

        topics = res.json().get("topic_list", {}).get("topics", [])
        if not topics:
            break

        for topic in topics:
            created = datetime.strptime(topic["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
            if START_DATE <= created <= END_DATE:
                filtered.append(topic)

        page += 1

    return filtered


# ---- FETCH FULL TOPIC CONTENT ----
def fetch_raw_html(topic_id, slug):
    url = f"{BASE_URL}/t/{slug}/{topic_id}.json"
    res = requests.get(url, cookies=cookies)
    if res.status_code != 200:
        return None

    topic_data = res.json()
    posts = topic_data.get("post_stream", {}).get("posts", [])

    return {
        "topic_id": topic_id,
        "title": topic_data.get("title"),
        "created_at": topic_data.get("created_at"),
        "posts": [
            {
                "post_number": post["post_number"],
                "username": post["username"],
                "created_at": post["created_at"],
                "raw_html": post["cooked"],
            }
            for post in posts
        ],
    }


# ---- MAIN ----
topics = get_filtered_topics()
all_data = []

for topic in topics:
    print(f"Fetching topic {topic['id']} - {topic['title']}")
    data = fetch_raw_html(topic["id"], topic["slug"])
    if data:
        all_data.append(data)
    time.sleep(0.5)

# ---- SAVE ----
with open("tds_raw_html.json", "w") as f:
    json.dump(all_data, f, indent=2)

print("✅ Done. Saved raw HTML for all filtered topics.")
