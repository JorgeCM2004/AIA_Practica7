import os

import pandas as pd
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, HTTPException
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from utils import Hybrid_Searcher
from utils.agent import Maternal_Agent, Medical_Agent
from utils.logger.F_logger import logger
from utils.tabular.F_Data_Preprocessor import Data_Preprocessor
from utils.tabular.F_XGBoost import XGBoost
from utils.vision.F_CNN_Classifier import CNN_Classifier
from utils.vision.F_UNet_Model import UNet_Model
from utils.tabular.F_Explainer import Explainer

app = FastAPI(title="Maternal Health AI Server", version="1.0")
searcher = Hybrid_Searcher()
paciente_agent = Maternal_Agent(searcher=searcher)
medico_agent = Medical_Agent()
cnn_model = CNN_Classifier(weights_path="weights/CNN_Classifier.pth")
xgb_model = XGBoost()
preprocessor = Data_Preprocessor()
explainer = Explainer()
unet = UNet_Model()

class PacienteRequest(BaseModel):
	query: str
	session_id: str


class MedicoRequest(BaseModel):
	query: str
	uploaded_files: list[str] = []


class TabularRequest(BaseModel):
	patient_data_csv: str


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

		respuesta = medico_agent.run(
			query=request.query,
			uploaded_files=request.uploaded_files,
			callbacks=[lf_handler],
		)

		logger.info(f"<medical agent response> {respuesta}")
		return {"respuesta": respuesta}
	except Exception as e:
		logger.error(f"<ERROR from doctor agent> {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/medico/tabular")
def predict_from_tabular_data(payload: TabularRequest):
	try:
		patient_data_csv = payload.patient_data_csv
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

		explicaciones_lista = explainer.explain(xgb_model.model, df_escalado, predicciones)

		datos_explicacion = explicaciones_lista[0]

		lista_razones = []
		for feature in datos_explicacion["top_features"]:
			lista_razones.append(f"{feature['feature']} ({feature['direction']} con impacto {feature['shap_value']:.2f})")

		explicacion = ", ".join(lista_razones)
		return f"ML RESULT: XGBoost prediction is Class {prediccion_cruda} based on the provided metrics. The SHAP explanation is {explicacion}"

	except Exception as e:
		return f"Error during ML Tabular inference: {str(e)}"


@app.post("/api/medico/vision")
def analyze_ultrasound(filename: str = "current_scan.png") -> str:
    ruta_fija = f"temp_uploads/{filename}"

    if not os.path.exists(ruta_fija):
        return f"Error: Image '{filename}' not found. Ask the doctor to upload an ultrasound."

    try:
        diagnostico_cnn = cnn_model.predict(ruta_fija)
        img_original = Image.open(ruta_fija).convert("RGB")
        img_gray = img_original.convert("L")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        input_tensor = transform(img_gray).unsqueeze(0).to(unet.device)

        unet.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=unet.device.type):
                output = unet.model(input_tensor)

            mask_tensor = (output > 0.5).float().cpu()[0][0]

        mask_np = mask_tensor.numpy()

        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_original.size, Image.NEAREST)

        mask_np_resized = np.array(mask_img)
        img_np = np.array(img_original)
        contours, _ = cv2.findContours(mask_np_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contours, -1, (255, 0, 0), thickness=2)
        img_final = Image.fromarray(img_np)

        os.makedirs("temp_results", exist_ok=True)
        nombre_resultado = f"segmented_{filename}"
        ruta_guardado = f"temp_results/{nombre_resultado}"
        img_final.save(ruta_guardado)

        return (
            f"CV RESULT:\n"
            f"- CNN Tissue Classification: {diagnostico_cnn}\n"
            f"- Model: UNet Semantic Segmentation\n"
            f"Visual evidence ready. IMPORTANT: You must include this exact tag in your response: [IMG]{ruta_guardado}[/IMG]"
        )

    except Exception as e:
        return f"Error during Vision inference (UNet): {str(e)}"


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
