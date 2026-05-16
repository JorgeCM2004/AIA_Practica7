import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional


class CNN_Structure(nn.Module):
	def __init__(
		self,
	):
		super().__init__()
		self.network = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten(),
			nn.Linear(64 * 28 * 28, 128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128, 2),
		)

	def forward(self, x):
		x = self.network(x)
		return x


class CNN_Classifier:
	def __init__(self, weights_path="weights/CNN_Classifier.pth"):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = CNN_Structure().to(self.device)

		if os.path.exists(weights_path):
			self.model.load_state_dict(
				torch.load(weights_path, map_location=self.device, weights_only=True)
			)

		self.model.eval()

		self.transform = transforms.Compose(
			[
				transforms.Lambda(self._fill_image),
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5], std=[0.5]),
			]
		)
		self.clases = {0: "BENIGN", 1: "MALIGNANT"}

	def _fill_image(self, image):
		w, h = image.size
		max_wh = max(w, h)
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return functional.pad(image, padding, 0, "constant")

	def predict(self, image_path: str) -> str:
		try:
			image = Image.open(image_path).convert("L")
			tensor = self.transform(image).unsqueeze(0).to(self.device)

			with torch.no_grad():
				outputs = self.model(tensor)
				_, predicted = torch.max(outputs, 1)
				class_idx = predicted.item()

			diagnostico = self.clases.get(class_idx, "UNKNOWN")
			return diagnostico
		except Exception as e:
			return e
