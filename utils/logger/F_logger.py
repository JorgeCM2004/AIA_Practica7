import logging
import os
from logging.handlers import RotatingFileHandler

RUTA_LOGS = "/app/logs"
os.makedirs(RUTA_LOGS, exist_ok=True)

logger = logging.getLogger("Medical_Copilot")
logger.setLevel(logging.INFO)

formato = logging.Formatter(
	"%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

ruta_archivo = os.path.join(RUTA_LOGS, "backend.log")
file_handler = RotatingFileHandler(
	ruta_archivo, maxBytes=5000000, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(formato)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formato)

if not logger.handlers:
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)
