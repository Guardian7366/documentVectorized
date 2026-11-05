from recommender_embeddings import EmbeddingRecommender

# Cargar documentos y generar embeddings
recommender = EmbeddingRecommender()

# Obtener los cinco documentos similares a DOC001
results = recommender.recommend_by_id("DOC001")

for result in results:
    print(result)
