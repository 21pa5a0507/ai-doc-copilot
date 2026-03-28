from crawl4ai import AsyncWebCrawler
from urllib.parse import urljoin, urlparse
import asyncio
from collections import deque
from bs4 import BeautifulSoup
import json
import re
import requests

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
def chunk_by_headings(main_tag, page_title=""):
    chunks = []

    current_heading = page_title
    current_content = []

    for tag in main_tag.find_all(["h1", "h2", "h3", "p", "li"]):

        if tag.name in ["h1", "h2", "h3"]:
            if current_content:
                chunks.append({
                    "title": current_heading,
                    "content": " ".join(current_content)
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
                    chunks = chunk_by_headings(main, title)

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
        "fastapi", "release notes"
    ]

    if any(word in text_lower for word in blacklist):
        return False

    return True

def chunking_docs(docs):
    chunked_data = []

    seen = set()   # deduplication

    for doc in docs:

        chunks = chunk_text(doc["content"])

        for chunk in chunks:
            if not is_valid_chunk(chunk):
                continue

            key = (doc["url"], chunk)
            if key in seen:
                continue
            seen.add(key)

            chunked_data.append({
                "url": doc["url"],
                "title": doc["title"],
                "content": chunk
            })
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

async def scrap_website(store):
    print("🚀 Getting initial URLs...")
    start_urls = get_all_urls()

    print(f"✅ Initial URLs: {len(start_urls)}")

    print(f"🚀 Crawling with depth={MAX_DEPTH}...")
    docs = await crawl_with_depth(start_urls)

    print(f"✅ Total chunks: {len(docs)}")

    with open("fastapi_docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    # with open("fastapi_docs.json", "r", encoding="utf-8") as f:
    #     docs = json.load(f)

    chunked_docs = chunking_docs(docs)
    store = embedding_docs(chunked_docs, store)
    store.build_bm25()
    return store

# def extract_links_from_html(html):
#     """Extract all href links from HTML"""
#     if not html:
#         return []
#     # Find all href attributes
#     links = re.findall(r'href=["\'](.*?)["\']', html)
#     return links

# async def scrap_website():
#     async with AsyncWebCrawler() as crawler:

#         while queue:
#             url, depth = queue.popleft()
#             url = normalize_url(url)
#             if depth > MAX_DEPTH:
#                 continue
#             if url in visited_urls:
#                 continue
#             visited_urls.add(url)
#             try:
#                 print(f"\n🔄 Crawling: {url} (Depth: {depth})")
#                 result = await crawler.arun(url=url, crawl_links=True)
#                 docs.append({
#                     "url":url,
#                     "content":result.markdown
#                 })
                
#                 # Try multiple ways to get links
#                 all_links = set()
                
#                 # Method 1: From result.links
#                 if result.links:
#                     print(f"  ✓ Found {len(result.links)} links in result.links")
#                     all_links.update(result.links)
                
#                 # Method 2: From raw HTML
#                 if hasattr(result, 'html') and result.html:
#                     html_links = extract_links_from_html(result.html)
#                     print(f"  ✓ Found {len(html_links)} links in HTML")
#                     all_links.update(html_links)
                
#                 print(f"  Total unique links found: {len(all_links)}")
                
#                 for link in all_links:
#                     if not link:
#                         continue
#                     if isinstance(link, dict):
#                         href = link.get("href")
#                     else:
#                         href = link
                    
#                     if not href:
#                         continue
                    
#                     if not href.startswith(("http://", "https://")):
#                         continue
                    
#                     normalized_link = normalize_url(href)
#                     if normalized_link not in visited_urls:
#                         queue.append((normalized_link, depth+1))
                        
#             except Exception as e:
#                 print(f"❌ Error scraping {url}: {e}")
#         with open("mdm_docs.json","w") as f:
#             json.dump(docs, f, indent=2)
#         print(f"\n✅ Scraping complete! Saved {len(docs)} documents to mdm_docs.json")



        # raw_text = result.markdown
        # if not raw_text:
        #     print(raw_text)
        #     raw_text = " "
        
        # cleaned = clean_text(raw_text)

        # if len(cleaned) > 200:
        #     chunks = chunk_text(cleaned)
        # else:        
        #     chunks = chunk_text(cleaned)

        # store = store

        # for chunk in chunks:
        #     emb = get_embending(chunk)
        #     print(type(emb))
        #     print(len(emb))
        #     store.add(emb,chunk)
        
        # query = "what is python?"

        # query_embedding = get_embending(query)

        # results = store.search(query_embedding)

        # print("\nSearch Results:\n")

        # for r in results:
        #     print("-", r[:200])
        
        # return chunks


if __name__ == "__main__":
    asyncio.run(main())
    print("executed")

    # print("len of chunks is: "+str(len(chunks)))