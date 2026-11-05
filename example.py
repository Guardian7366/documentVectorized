from recommender_embeddings import EmbeddingRecommender

# Load documents and generate embeddings
recommender = EmbeddingRecommender()

# Get the five documents similar to DOC001
results = recommender.recommend_by_id("DOC001")

for result in results:
    print(result)
