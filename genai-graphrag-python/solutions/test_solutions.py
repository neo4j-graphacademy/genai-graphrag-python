def test_kg_structured_builder(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "kg_structured_builder",
    )

    result = test_helpers.run_cypher(
        "RETURN EXISTS {(:__KGBuilder__)} as exists"
        )
    
    assert result[0]["exists"]
    
    assert output > ""

def test_vector_cypher_rag(test_helpers, monkeypatch):

    create_index_cypher = """
    CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
    FOR (n:Chunk)
    ON n.embedding
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};
    """

    test_helpers.run_cypher(
        create_index_cypher
        )

    output = test_helpers.run_module(
        monkeypatch, 
        "vector_cypher_rag",
    )

    assert output > ""

def test_text2cypher_rag(test_helpers, monkeypatch):

    output = test_helpers.run_module(
        monkeypatch, 
        "text2cypher_rag",
    )

    assert output > ""
