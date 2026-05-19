import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from utils import Hybrid_Searcher
from utils.agent import Maternal_Agent, Medical_Agent
from utils.logger.F_logger import logger
from utils.tabular.F_Data_Preprocessor import Data_Preprocessor
from utils.tabular.F_XGBoost import XGBoost
from utils.vision.F_CNN_Classifier import CNN_Classifier
from utils.vision.F_YOLO_Model import YOLO_Model

app = FastAPI(title="Maternal Health AI Server", version="1.0")
searcher = Hybrid_Searcher()
paciente_agent = Maternal_Agent(searcher=searcher)
medico_agent = Medical_Agent()
cnn_model = CNN_Classifier(weights_path="weights/CNN_Classifier.pth")
yolo_model = YOLO_Model(model_path="weights/YOLO.pt")
xgb_model = XGBoost()
preprocessor = Data_Preprocessor()


class PacienteRequest(BaseModel):
	query: str
	session_id: str


class MedicoRequest(BaseModel):
	query: str


@app.post("/api/paciente/chat")
def chat_paciente(request: PacienteRequest):
	try:
		logger.info(f"<patient query> {request.query}")

		lf_handler = CallbackHandler()

		respuesta = paciente_agent.run(
			query=request.query,
			session_id=request.session_id,
			callbacks=[lf_handler],
		)

		logger.info(f"<maternal agent response> {respuesta}")
		return {"respuesta": respuesta}
	except Exception as e:
		logger.error(f"<ERROR from patient agent> {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/medico/chat")
def chat_medico(request: MedicoRequest):
	try:
		logger.info(f"<doctor query> {request.query}")

		lf_handler = CallbackHandler()

		respuesta = medico_agent.run(query=request.query, callbacks=[lf_handler])

		logger.info(f"<medical agent response> {respuesta}")
		return {"respuesta": respuesta}
	except Exception as e:
		logger.error(f"<ERROR from doctor agent> {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/medico/tabular")
def predict_from_tabular_data(patient_data_csv: str):
	try:
		valores = [float(x.strip()) for x in patient_data_csv.split(",")]

		if len(valores) != 21:
			return f"Error: Expected exactly 21 values, but received {len(valores)}. Please ask the doctor to provide all 21 metrics."

		columnas_esperadas = [
			"baseline value",
			"accelerations",
			"fetal_movement",
			"uterine_contractions",
			"light_decelerations",
			"severe_decelerations",
			"prolongued_decelerations",
			"abnormal_short_term_variability",
			"mean_value_of_short_term_variability",
			"percentage_of_time_with_abnormal_long_term_variability",
			"mean_value_of_long_term_variability",
			"histogram_width",
			"histogram_min",
			"histogram_max",
			"histogram_number_of_peaks",
			"histogram_number_of_zeroes",
			"histogram_mode",
			"histogram_mean",
			"histogram_median",
			"histogram_variance",
			"histogram_tendency",
		]

		df_paciente = pd.DataFrame([valores], columns=columnas_esperadas)

		df_escalado = preprocessor.preprocess(df_paciente)
		predicciones = xgb_model.predict(df_escalado)

		prediccion_cruda = predicciones[0]

		return f"ML RESULT: XGBoost prediction is Class {prediccion_cruda} based on the provided metrics."

	except Exception as e:
		return f"Error during ML Tabular inference: {str(e)}"


@app.post("/api/medico/vision")
def analyze_ultrasound() -> str:
	ruta_fija = "temp_uploads/current_scan.png"

	if not os.path.exists(ruta_fija):
		return "Error: No image uploaded yet. Ask the doctor to upload an ultrasound."

	try:
		diagnostico_cnn = cnn_model.predict(ruta_fija)
		coordenadas = yolo_model.predict(ruta_fija)
		img_marcada = yolo_model.draw_box(ruta_fija, coordenadas)

		os.makedirs("temp_results", exist_ok=True)
		ruta_guardado = "temp_results/marked_current_scan.png"
		img_marcada.save(ruta_guardado)

		return (
			f"CV RESULT:\n"
			f"- CNN Tissue Classification: {diagnostico_cnn}\n"
			f"- YOLO Bounding Box Coordinates: {coordenadas}\n"
			f"Visual evidence ready. IMPORTANT: You must include this exact tag in your response: [IMG]{ruta_guardado}[/IMG]"
		)
	except Exception as e:
		return f"Error during Vision inference: {str(e)}"


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
