import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# tag::import_splitter[]
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
# end::import_splitter[]

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# tag::splitter[]
class SectionSplitter(TextSplitter):
    def __init__(self, section_heading: str = "== ") -> None:
        self.section_heading = section_heading

    async def run(self, text: str) -> TextChunks:
        index = 0
        chunks = []
        current_section = ""

        for line in text.split('\n'):
            # Does the line start with the section heading?
            if line.startswith(self.section_heading):
                chunks.append(
                    TextChunk(text=current_section, index=index)
                )
                current_section = ""
                index += 1
            
            current_section += line + "\n"

        # Add the last section
        chunks.append(
            TextChunk(text=current_section, index=index)
        )
        
        return TextChunks(chunks=chunks)

splitter = SectionSplitter()
# end::splitter[]

# tag::run_splitter[]
text = """
= Heading 1
This is the main section

== Sub-heading
This is some text.

== Sub-heading 2
This is some more text.
"""

chunks = asyncio.run(splitter.run(text))
print(chunks)
# end::run_splitter[]

# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder,
    from_pdf=True,
    text_splitter=splitter,
)
# end::kg_builder[]

pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"

print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
