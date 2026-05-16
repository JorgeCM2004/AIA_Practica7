import os

from cryptography.fernet import Fernet
from dotenv import load_dotenv


class Encrypter:
	def __init__(self):
		load_dotenv()

		key = os.getenv("ENCRYPTION_KEY")

		if not key:
			raise ValueError("No se encontró 'ENCRYPTION_KEY' en el archivo .env")

		self.cipher_suite = Fernet(key.encode("utf-8"))

	def encrypt(self, plain_text: str) -> str:
		if not plain_text:
			return plain_text

		encoded_text = plain_text.encode("utf-8")
		encrypted_bytes = self.cipher_suite.encrypt(encoded_text)
		return encrypted_bytes.decode("utf-8")

	def decrypt(self, encrypted_text: str) -> str:
		if not encrypted_text:
			return encrypted_text

		encrypted_bytes = encrypted_text.encode("utf-8")
		decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
		return decrypted_bytes.decode("utf-8")
