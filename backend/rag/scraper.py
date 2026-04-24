from crawl4ai import AsyncWebCrawler
import asyncio
import json
import logging
import random
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config.paths import (
    HEXNODE_CHUNK_CACHE,
    HEXNODE_EMB_INDEX,
    HEXNODE_META_CACHE,
    HEXNODE_RAW_CACHE,
    LEGACY_HEXNODE_DOCS_JSON,
)
from rag.cleaner import clean_text
from rag.chunker import chunk_text
from rag.embeddings import get_embedding


logger = logging.getLogger(__name__)

SITEMAP_URL = "https://www.hexnode.com/mobile-device-management/help/category-sitemap.xml"
BASE_DOMAIN = "hexnode.com"
MAX_DEPTH = 3


# -----------------------------
# 1. GET INITIAL URLS
# -----------------------------
def get_all_urls():
    res = requests.get(SITEMAP_URL)
    soup = BeautifulSoup(res.text, "xml")

    urls = [loc.text.strip() for loc in soup.find_all("loc")]

    # Keep the crawl focused on the Windows docs.
    windows_urls = [
        url for url in urls
        if "windows" in url.lower()
    ]

    logger.info("Collected %s Windows Hexnode URLs from sitemap", len(windows_urls))
    return windows_urls


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
            "content": " ".join(current_content),
            "url": url,
        })

    return chunks


def normalize_url(url: str) -> str:
    url = url.split("#")[0]
    url = url.split("?")[0]
    return url.rstrip("/")


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

            # Small delay before each request.
            await asyncio.sleep(random.uniform(0.5, 1.5))

            try:
                result = await crawler.arun(url=url)

                if not result or not result.html:
                    continue

                html = result.html
                logger.info("Visiting %s at depth %s (%s pages visited)", url, depth, len(visited))

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
                    link = normalize_url(link)

                    if link not in visited and "windows" in link.lower():
                        queue.append((link, depth + 1))

            except Exception as exc:
                logger.warning("Failed to crawl %s: %s", url, exc)

    return all_docs

def is_valid_chunk(text):
    text_lower = text.lower().strip()

    # Exact junk phrases.
    exact_blacklist = [
        "next",
        "previous",
        "edit on github",
        "table of contents",
        "navigation",
    ]

    if text_lower in exact_blacklist:
        return False

    # Very short chunks usually add noise.
    if len(text_lower) < 30:
        return False

    # Too many symbols usually means menu or UI noise.
    symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if symbol_ratio > 0.3:
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

    logger.info("Prepared %s clean chunks after deduplication", len(chunked_data))
    return chunked_data


def load_json_cache(*paths):
    for candidate in paths:
        candidate = Path(candidate)
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def save_json_cache(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def embedding_docs(docs, store):
    texts = [doc["content"] for doc in docs]

    logger.info("Generating embeddings for %s chunks", len(docs))
    embeddings = get_embedding(texts)

    for doc, emb in zip(docs, embeddings):
        store.add(emb, doc)

    logger.info("Embeddings stored")

    return store


async def scrap_website(store, raw_cache=HEXNODE_RAW_CACHE, chunk_cache=HEXNODE_CHUNK_CACHE):
    docs = load_json_cache(raw_cache, LEGACY_HEXNODE_DOCS_JSON)
    if docs is not None:
        logger.info("Loaded %s Hexnode raw documents from cache", len(docs))
    else:
        logger.info("Fetching initial Hexnode URLs")
        start_urls = get_all_urls()

        logger.info("Starting crawl with %s initial URLs", len(start_urls))

        logger.info("Crawling Hexnode docs with max depth %s", MAX_DEPTH)
        docs = await crawl_with_depth(start_urls)
        save_json_cache(raw_cache, docs)

    chunked_docs = load_json_cache(chunk_cache)
    if chunked_docs is not None:
        logger.info("Loaded %s Hexnode chunks from cache", len(chunked_docs))
    else:
        chunked_docs = chunking_docs(docs)
        save_json_cache(chunk_cache, chunked_docs)

    embedding_docs(chunked_docs, store)
    store.build_bm25()  # Build BM25 after all chunks added
    store.save(HEXNODE_EMB_INDEX, meta_path=HEXNODE_META_CACHE)  # Save after embedding

    return True


if __name__ == "__main__":
    print("Use initialize_vector_store() or await scrap_website(store) to build the Hexnode index.")
