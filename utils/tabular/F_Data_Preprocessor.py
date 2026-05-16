import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Data_Preprocessor:
	def __init__(self):
		self.std_scaler: StandardScaler = None
		self.label_encoder: LabelEncoder = None
		self._load_preprocessor()

	def _load_preprocessor(self, load_dir="weights"):
		file_path = os.path.join(load_dir, "Data_Preprocessor.joblib")
		if os.path.exists(file_path):
			data = joblib.load(file_path)
			self.std_scaler = data["std_scaler"]
			self.label_encoder = data["label_encoder"]

	def preprocess(self, data: pd.DataFrame, columns=None) -> pd.DataFrame:
		if not columns:
			columns = data.columns
		data[columns] = self.std_scaler.transform(data[columns])
		return data

	def decode(self, encoded_labels):
		return self.label_encoder.inverse_transform(encoded_labels)
