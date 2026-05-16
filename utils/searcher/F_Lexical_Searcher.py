from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Lexical_Searcher:
	def __init__(self):
		self.base_dir = Path(__file__).parent.parent.parent
		self.data_path = self.base_dir / "data" / "ghana_maternal_health.csv"

		self.df = pd.read_csv(self.data_path)

		self.corpus = (
			"Patient Symptom/Query: "
			+ self.df["Question"]
			+ "\nMedical Action: "
			+ self.df["Answer"]
		)

		self.vectorizer = TfidfVectorizer(stop_words="english")
		self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

	def search(self, user_query, top_k=3):
		query_vec = self.vectorizer.transform([user_query])

		cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

		top_indices = cosine_sim.argsort()[-top_k:][::-1]

		documents = [self.corpus.iloc[i] for i in top_indices]
		scores = [cosine_sim[i] for i in top_indices]

		return {"documents": [documents], "scores": [scores]}
