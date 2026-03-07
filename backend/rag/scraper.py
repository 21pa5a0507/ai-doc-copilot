from crawl4ai import AsyncWebCrawler
import asyncio
from cleaner import clean_text
from chunker import chunk_text
from embendings import get_embending

async def scrap_website(url: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url = url)
        
        raw_text = result.markdown

        cleaned = clean_text(raw_text)

        chunks = chunk_text(cleaned)
        
        return chunks

if __name__ == "__main__":
    url ="https://docs.python.org/3/tutorial/introduction.html"

    chunks = asyncio.run(scrap_website(url))

    print("len of chunks is: "+str(len(chunks)))
    print(chunks[0])
    print(type(chunks[0][0]))
    vector = get_embending(chunks[0])
    print(len(vector))
    print(vector)
