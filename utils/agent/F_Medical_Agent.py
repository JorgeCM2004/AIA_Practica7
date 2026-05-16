import os
from typing import Annotated

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from utils.tabular.F_Data_Preprocessor import Data_Preprocessor
from utils.tabular.F_XGBoost import XGBoost
from utils.vision.F_CNN_Classifier import CNN_Classifier
from utils.vision.F_YOLO_Model import YOLO_Model


class AgentState(TypedDict):
	messages: Annotated[list, add_messages]


class Medical_Agent:
	def __init__(self):
		load_dotenv()

		self.cnn_model = CNN_Classifier(weights_path="weights/CNN_Classifier.pth")
		self.yolo_model = YOLO_Model(model_path="weights/YOLO.pt")

		self.xgb_model = XGBoost()
		self.preprocessor = Data_Preprocessor()

		@tool
		def analyze_ultrasound() -> str:
			"""
			Use this tool to analyze medical ultrasounds.
			Call this tool immediately when the doctor asks to see or analyze the uploaded image.
			"""
			ruta_fija = "temp_uploads/current_scan.png"

			if not os.path.exists(ruta_fija):
				return "Error: No image uploaded yet. Ask the doctor to upload an ultrasound."

			try:
				diagnostico_cnn = self.cnn_model.predict(ruta_fija)
				coordenadas = self.yolo_model.predict(ruta_fija)
				img_marcada = self.yolo_model.draw_box(ruta_fija, coordenadas)

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

		@tool
		def predict_from_tabular_data(patient_data_csv: str) -> str:
			"""
			Use this tool to calculate risks based on tabular CTG data.
			You must pass a single comma-separated string of exactly 21 numbers.
			"""
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

				df_escalado = self.preprocessor.preprocess(df_paciente)
				predicciones = self.xgb_model.predict(df_escalado)

				prediccion_cruda = predicciones[0]

				return f"ML RESULT: XGBoost prediction is Class {prediccion_cruda} based on the provided metrics."

			except Exception as e:
				return f"Error during ML Tabular inference: {str(e)}"

		self.tools = [analyze_ultrasound, predict_from_tabular_data]

		self.llm_base = ChatGroq(
			model="openai/gpt-oss-120b",
			temperature=0,
			api_key=os.getenv("GROQ_API_KEY"),
		)
		self.llm = self.llm_base.bind_tools(self.tools)

		self.agent = self._build_graph()

	def agent_node(self, state: AgentState):
		response = self.llm.invoke(state["messages"])
		return {"messages": [response]}

	def _build_graph(self):
		workflow = StateGraph(AgentState)
		workflow.add_node("agent", self.agent_node)
		workflow.add_node("tools", ToolNode(self.tools))

		workflow.add_edge(START, "agent")
		workflow.add_conditional_edges("agent", tools_condition)
		workflow.add_edge("tools", "agent")

		return workflow.compile()

	def run(self, query: str):
		sys_msg = SystemMessage(
			content="""You are an Advanced Clinical Copilot designed exclusively to assist Obstetricians.
            You have access to predictive AI tools (Computer Vision and Machine Learning).

            CRITICAL INSTRUCTIONS:
            1. You are a technical assistant. Use precise medical and scientific language.
            2. If the doctor asks you to analyze an image, use the 'analyze_ultrasound' tool immediately.
            3. CTG DATA: If the doctor provides numbers for a Cardiotocogram (CTG), extract them in the exact order provided.
            4. FORBIDDEN: Do not interpret the numbers yourself.
            5. YOU MUST IMMEDIATELY use the 'predict_from_tabular_data' tool. Pass the 21 numbers as a single comma-separated string to the 'patient_data_csv' parameter (e.g., "133.0, 0.002, 0.01").
            6. ONLY AFTER the tool returns the XGBoost result, integrate it into your response and suggest clinical steps.
            7. IF THE 'analyze_ultrasound' TOOL RETURNS AN [IMG]...[/IMG] TAG, YOU MUST INCLUDE THAT EXACT TAG IN YOUR RESPONSE.
            """
		)

		inputs = {"messages": [sys_msg, HumanMessage(content=query)]}
		final_state = self.agent.invoke(inputs)
		return final_state["messages"][-1].content
