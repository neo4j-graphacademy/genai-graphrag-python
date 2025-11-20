import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    TextChunks,
    TextChunk
)
from neo4j_graphrag.llm import OpenAILLM

text = """
Neo4j is a graph database that stores data in a graph structure.
Data is stored as nodes and relationships instead of tables or documents.
Graph databases are particularly useful when _the connections between data are as important as the data itself_.
A graph shows how objects are related to each other.
The objects are referred to as *nodes* (vertices) connected by *relationships* (edges).
Neo4j uses the graph structure to store data and is known as a *labeled property graph*.
"""

extractor = LLMEntityRelationExtractor(
    llm=OpenAILLM(
        model_name="gpt-4",
        model_params={"temperature": 0}
    )
)

entities = asyncio.run(
    extractor.run(
        chunks=TextChunks(chunks=[TextChunk(text=text, index=0)])
    )
)

print(entities)
