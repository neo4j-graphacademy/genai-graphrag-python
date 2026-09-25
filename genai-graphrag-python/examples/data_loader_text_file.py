import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# tag::import_loader[]
from neo4j_graphrag.components.data_loader import DataLoader
from neo4j_graphrag.components.types import LoadedDocument, DocumentInfo
from pathlib import Path
from typing import Dict, Optional
# end::import_loader[]

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-5-nano",
    model_params={
        "reasoning_effort": "minimal"
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# tag::loader[]
class TextLoader(DataLoader):
    async def run(self, filepath: Path, metadata: Optional[Dict[str, str]] = None) -> LoadedDocument:

        # Process the file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Return a LoadedDocument
        return LoadedDocument(
            text=text,
            document_info=DocumentInfo(
                path=str(filepath),
                metadata=metadata or {}
            )
        )
    
data_loader = TextLoader()
# end::loader[]

# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_file=True,
    file_loader=data_loader
)
# end::kg_builder[]

# tag::run_loader[]
pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.txt"
doc = asyncio.run(data_loader.run(pdf_file))
print(doc.text)
# end::run_loader[]

pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.txt"
print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
