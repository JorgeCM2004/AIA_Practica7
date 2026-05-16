import os

from PIL import Image, ImageDraw
from ultralytics import YOLO


class YOLO_Model:
	def __init__(self, model_path="weights/YOLO.pt"):
		if os.path.exists(model_path):
			self.model = YOLO(model_path)
		else:
			raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

	def predict(self, image_path: str) -> str:
		try:
			results = self.model.predict(image_path, conf=0.000001, verbose=False)

			if len(results[0].boxes) > 0:
				best_box = results[0].boxes[0]

				x, y, w, h = best_box.xywhn[0].tolist()

				return (x, y, w, h)
			else:
				return (0, 0, 0, 0)

		except Exception as e:
			return e

	def draw_box(self, image_path: str, bounding_boxes: tuple) -> Image.Image:
		try:
			imagen = Image.open(image_path).convert("RGB")

			if (
				bounding_boxes == (0, 0, 0, 0)
				or bounding_boxes[2] == 0
				or bounding_boxes[3] == 0
			):
				return imagen

			x, y, w, h = bounding_boxes
			ancho_img, alto_img = imagen.size

			centro_x = x * ancho_img
			centro_y = y * alto_img
			ancho_caja = w * ancho_img
			alto_caja = h * alto_img

			x0 = centro_x - (ancho_caja / 2)
			y0 = centro_y - (alto_caja / 2)
			x1 = centro_x + (ancho_caja / 2)
			y1 = centro_y + (alto_caja / 2)

			draw = ImageDraw.Draw(imagen)
			draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

			return imagen

		except Exception:
			return Image.open(image_path).convert("RGB")
