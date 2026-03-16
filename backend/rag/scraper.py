from crawl4ai import AsyncWebCrawler
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
docs = []

def get_urls_from_sitemap():
    """Fetch all documentation URLs from sitemap"""

    print("🔎 Fetching sitemap...")

    response = requests.get(SITEMAP_URL)
    soup = BeautifulSoup(response.text, "xml")

    urls = []

    for loc in soup.find_all("loc"):
        urls.append(loc.text.strip())

    print(f"✅ Found {len(urls)} URLs")

    return urls

def normalize_url(url):
    url = url.split("?")[0]
    return url.rstrip("/")

async def scrape_page(crawler, url):
    """Scrape a single page"""

    try:
        print(f"🔄 Scraping: {url}")

        result = await crawler.arun(url=url)

        if not result:
            return None

        content = result.markdown

        if not content or len(content) < 200:
            return None

        return {
            "url": url,
            "title": result.metadata.get("title") if result.metadata else "",
            "content": clean_text(content)
        }

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

async def scrape_all(urls):

    docs = []

    async with AsyncWebCrawler() as crawler:

        tasks = [
            scrape_page(crawler, url)
            for url in urls
        ]

        results = await asyncio.gather(*tasks)

        for r in results:
            if r:
                docs.append(r)

    return docs

""" <================= Chunking docs =================> """
def chunking_docs(docs):
    chunked_data = []

    for doc in docs:

        chunks = chunk_text(doc["content"])

        for chunk in chunks:

            chunked_data.append({
                "url": doc["url"],
                "title": doc["title"],
                "content": chunk
            })
    return chunked_data

""" <================= End of chunking docs =================> """

""" <================= Embedding docs =================> """

def embedding_docs(docs,store=None):
    for doc in docs:
        if doc["content"] and len(doc["content"]) > 0:
            emb = get_embending(doc["content"])
            store.add(emb, doc)

    return store

async def scrap_website(store):

    # urls = get_urls_from_sitemap()

    # docs = await scrape_all(urls)

    with open("fastapi_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunked_docs = chunking_docs(docs)
    search_store = embedding_docs(chunked_docs, store)
    store.update_bm25()
    return search_store

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