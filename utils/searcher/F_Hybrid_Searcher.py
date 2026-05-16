from .F_Lexical_Searcher import Lexical_Searcher
from .F_Semantic_Searcher import Semantic_Searcher


class Hybrid_Searcher:
	def __init__(self):
		self.semantic_searcher = Semantic_Searcher()
		self.lexical_searcher = Lexical_Searcher()

	def search(self, user_query, top_k=3, lexical_percentage=0.5, batch_retrieval=None):
		pool_retrieval = 50 if not batch_retrieval else batch_retrieval
		semantic_results = self.semantic_searcher.search(
			user_query, top_k=pool_retrieval
		)
		lexical_results = self.lexical_searcher.search(user_query, top_k=pool_retrieval)

		semantic_docs = semantic_results["documents"][0]
		lexical_docs = lexical_results["documents"][0]

		fusion_scores = {}

		for rank, doc in enumerate(semantic_docs):
			fusion_scores[doc] = fusion_scores.get(doc, 0.0) + (
				1 - lexical_percentage
			) * (1.0 / (rank + 1))

		for rank, doc in enumerate(lexical_docs):
			fusion_scores[doc] = fusion_scores.get(doc, 0.0) + lexical_percentage * (
				1.0 / (rank + 1)
			)

		ranked_results = sorted(
			fusion_scores.items(), key=lambda item: item[1], reverse=True
		)

		final_docs = [item[0] for item in ranked_results[:top_k]]
		final_scores = [item[1] for item in ranked_results[:top_k]]

		return {"documents": [final_docs], "scores": [final_scores]}
