import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class AgentState(TypedDict):
	messages: Annotated[list, add_messages]


class Medical_Agent:
	def __init__(self):
		load_dotenv()
		self.api = os.getenv("API_URL", "http://127.0.0.1:8000")
		self._uploaded_files: list[str] = []

		@tool
		def analyze_ultrasound() -> str:
			"""
			Use this tool to analyze all uploaded medical ultrasound images.
			Call this tool immediately when the doctor asks to see or analyze
			the uploaded image(s). It processes every uploaded scan one by one
			and returns the combined results.
			"""
			if not self._uploaded_files:
				return "Error: No images have been uploaded yet. Ask the doctor to upload at least one ultrasound."

			resultados = []
			for filename in self._uploaded_files:
				try:
					url = f"{self.api}/api/medico/vision"
					respuesta = requests.post(url, params={"filename": filename})
					respuesta.raise_for_status()
					texto = respuesta.json()
					resultados.append(f"[Image: {filename}]\n{texto}")
				except Exception as e:
					resultados.append(f"[Image: {filename}]\nError: {str(e)}")

			return "\n\n".join(resultados)

		@tool
		def predict_from_tabular_data(patient_data_csv: str) -> str:
			"""
			Use this tool to calculate risks based on tabular CTG data.
			You must pass a single comma-separated string of exactly 21 numbers.
			"""
			payload = {"patient_data_csv": patient_data_csv}

			try:
				respuesta = requests.post(
					f"{self.api}/api/medico/tabular", json=payload
				)
				respuesta.raise_for_status()
				datos = respuesta.json()

				return f"El modelo de Machine Learning devolvió: {datos}"

			except Exception as e:
				return f"Error al contactar con el microservicio predictivo: {str(e)}"

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

	def run(self, query: str, uploaded_files: list[str] | None = None, callbacks=None):
		self._uploaded_files = uploaded_files or []

		sys_msg = SystemMessage(
			content="""
            [ROLE]
            You are an Advanced Clinical Copilot designed exclusively to assist Obstetricians.

            [AUDIENCE]
            Your primary users are medical professionals. You must use precise medical and scientific language at all times. Do not use layman's terms.

            [TASK]
            Your core objective is to process clinical inputs (medical imaging and tabular data) by routing them to the correct predictive AI tools and presenting the diagnostic results safely to the doctor.

            [EXECUTION & EXAMPLES]
            - ULTRASOUND: If the doctor asks you to analyze an image or mentions imaging, use the 'analyze_ultrasound' tool immediately. It will process ALL uploaded scans and return combined results.
            - CTG DATA: If the doctor provides numbers for a Cardiotocogram (CTG), extract them in the exact order provided. YOU MUST IMMEDIATELY use the 'predict_from_tabular_data' tool. Pass the 21 numbers as a single comma-separated string to the 'patient_data_csv' parameter (e.g., "133.0, 0.002, 0.01").

            [ACTION & CONSTRAINTS]
            - FORBIDDEN: Do not interpret the raw CTG numbers yourself under any circumstances.
            - ONLY AFTER the 'predict_from_tabular_data' tool returns the XGBoost result, integrate it into your response and suggest appropriate clinical steps.
            - MULTIMODAL RULE: If the 'analyze_ultrasound' tool returns one or more [IMG]...[/IMG] tags, YOU MUST INCLUDE ALL OF THEM in your final text response, each on its own line.
            """
		)

		inputs = {"messages": [sys_msg, HumanMessage(content=query)]}
		config = {"tags": ["doctor", "produccion"]}
		if callbacks:
			config["callbacks"] = callbacks
		final_state = self.agent.invoke(inputs, config=config)
		return final_state["messages"][-1].content
