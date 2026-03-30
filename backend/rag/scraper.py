from crawl4ai import AsyncWebCrawler
from urllib.parse import urljoin, urlparse
import asyncio
from collections import deque
from bs4 import BeautifulSoup
import json
import re
import requests
from pathlib import Path
from rag.cleaner import clean_text
from rag.chunker import chunk_text
from rag.embendings import get_embending
from rag.vector_store import VectorStore

SITEMAP_URL = "https://fastapi.tiangolo.com/sitemap.xml"
BASE_DOMAIN = "fastapi.tiangolo.com"
MAX_DEPTH = 3


# -----------------------------
# 1. GET INITIAL URLS
# -----------------------------
def get_all_urls():
    res = requests.get(SITEMAP_URL)
    soup = BeautifulSoup(res.text, "xml")

    return [loc.text.strip() for loc in soup.find_all("loc")]


# -----------------------------
# 2. EXTRACT LINKS FROM PAGE
# -----------------------------
def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if BASE_DOMAIN in parsed.netloc:
            clean_url = full_url.split("#")[0].rstrip("/")
            links.add(clean_url)

    return links


# -----------------------------
# 3. EXTRACT MAIN CONTENT
# -----------------------------
def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main") or soup.find("article")
    if not main:
        return None

    for tag in main.find_all(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    return main


# -----------------------------
# 4. HEADING CHUNKING
# -----------------------------
def chunk_by_headings(main_tag, page_title="", url=""):
    chunks = []

    current_heading = page_title
    current_content = []

    for tag in main_tag.find_all(["h1", "h2", "h3", "p", "li"]):

        if tag.name in ["h1", "h2", "h3"]:
            if current_content:
                chunks.append({
                    "title": current_heading,
                    "content": " ".join(current_content),
                    "url": url
                })
                current_content = []

            current_heading = tag.get_text(strip=True)

        else:
            text = tag.get_text(strip=True)
            if text:
                current_content.append(text)

    if current_content:
        chunks.append({
            "title": current_heading,
            "content": " ".join(current_content)
        })

    return chunks

def normalize_url(url: str) -> str:
    url = url.split("#")[0]
    url = url.split("?")[0]
    return url.rstrip("/")

def is_valid_url(url: str) -> bool:
    # Normalize quick checks
    url_lower = url.lower()

    # ❌ Skip obvious junk
    if any(x in url_lower for x in [
        "#", "mailto:", "javascript:", "?",
        "twitter", "facebook", "linkedin"
    ]):
        return False

    # ❌ Block unwanted sections
    blocked_paths = [
        "/search",
        "/tag",
        "/category",
        "/blog",
        "/news",
        "/release-notes",
        "/sponsors",
        "/help",
    ]

    if any(path in url_lower for path in blocked_paths):
        return False

    # ✅ Allow only real documentation sections
    allowed_paths = [
        "/tutorial/",
        "/advanced/",
        "/deployment/",
        "/python-types/",
        "/async/",
        "/security/",
    ]

    return any(path in url_lower for path in allowed_paths)
# -----------------------------
# 5. BFS CRAWLER WITH DEPTH
# -----------------------------
async def crawl_with_depth(start_urls):
    visited = set()
    queue = deque([(url, 0) for url in start_urls])

    all_docs = []

    async with AsyncWebCrawler() as crawler:

        while queue:
            url, depth = queue.popleft()

            if url in visited or depth > MAX_DEPTH:
                continue

            visited.add(url)

            try:
                result = await crawler.arun(url=url)

                if not result or not result.html:
                    continue

                html = result.html

                # -------- CONTENT --------
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else ""

                main = extract_main_content(html)
                if main:
                    chunks = chunk_by_headings(main, title, url)

                    for c in chunks:
                        content = clean_text(c["content"])
                        if len(content) > 50:
                            all_docs.append({
                                "url": url,
                                "title": c["title"],
                                "content": content
                            })

                # -------- LINKS --------
                links = extract_links(html, url)

                for link in links:
                    print(f"Visiting: {url} | Depth: {depth} | Total: {len(visited)}")
                    link = normalize_url(link)

                    if link not in visited and is_valid_url(link):
                        queue.append((link, depth + 1))

            except Exception as e:
                print(f"❌ Error: {url} -> {e}")

    return all_docs

""" <================= Chunking docs =================> """

def is_valid_chunk(text):
    text_lower = text.lower()

    blacklist = [
        "next", "previous", "edit on github",
        "table of contents", "navigation",
    ]

    if any(word in text_lower for word in blacklist):
        return False

    return True

def chunking_docs(docs):
    chunked_data = []

    seen = set()   # deduplication
    for doc in docs:
        cleaned = clean_text(doc["content"])
        sub_chunks = chunk_text(cleaned)

        for chunk in sub_chunks:
            if not is_valid_chunk(chunk):
                continue

            # Stable dedup key (url + content hash, not object id)
            key = doc["url"] + "||" + chunk
            if key in seen:
                continue
            seen.add(key)

            chunked_data.append({
                "url": doc["url"],
                "title": doc["title"],
                "content": chunk,
            })

    print(f"✅ Total clean chunks after dedup: {len(chunked_data)}")
    return chunked_data

""" <================= End of chunking docs =================> """

""" <================= Embedding docs =================> """

def embedding_docs(docs, store):

    texts = [doc["content"] for doc in docs]

    print("🔄 Generating embeddings in batch...")
    embeddings = get_embending(texts)   # MUST support batch

    for doc, emb in zip(docs, embeddings):
        store.add(emb, doc)

    print("✅ Embeddings stored")

    return store

async def scrap_website(store, json_cache = "fastapi_docs.json"):
    cache_path = Path(json_cache)

    if cache_path.exists():
         with open(cache_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
         print(f"✅ Loaded {len(docs)} documents from cache.")
    else:
        docs = []

        print("🚀 Getting initial URLs...")
        start_urls = get_all_urls()

        print(f"✅ Initial URLs: {len(start_urls)}")

        print(f"🚀 Crawling with depth={MAX_DEPTH}...")
        docs = await crawl_with_depth(start_urls)

        print(f"✅ Total chunks: {len(docs)}")

        with open("fastapi_docs.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)



    chunked_docs = chunking_docs(docs)
    store = embedding_docs(chunked_docs, store)
    store.build_bm25()
    return store


if __name__ == "__main__":
    asyncio.run(main())
    print("executed")

    # print("len of chunks is: "+str(len(chunks)))