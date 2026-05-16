import os

import joblib


class XGBoost:
	def __init__(self):
		self._load_model()

	def predict(self, x):
		y_hat = self.model.predict(x)
		return y_hat

	def _load_model(self, load_dir="weights"):
		file_path = os.path.join(load_dir, "XGBoost.joblib")

		if os.path.exists(file_path):
			self.model = joblib.load(file_path)
