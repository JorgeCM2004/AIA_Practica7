import os
import re
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="AIA_P7", layout="wide")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

if "session_id" not in st.session_state:
	st.session_state.session_id = str(uuid.uuid4())
if "mensajes_paciente" not in st.session_state:
	st.session_state.mensajes_paciente = []


def renderizar_mensaje_multimodal(texto):
	partes = re.split(r"(\[IMG\].*?\[/IMG\])", texto)
	for parte in partes:
		if parte.startswith("[IMG]") and parte.endswith("[/IMG]"):
			ruta_imagen = parte[5:-6]
			if os.path.exists(ruta_imagen):
				st.image(
					ruta_imagen,
					use_container_width=True,
				)
			else:
				st.warning("⚠️ La imagen procesada no se encontró en el servidor.")
		else:
			if parte.strip():
				st.write(parte)

headers = st.context.headers if hasattr(st.context, "headers") else {}

grupos_usuario = headers.get("X-authentik-groups", "").lower()

if "doctores" in grupos_usuario:
	st.title("Dashboard Clínico")

	col1, col2 = st.columns([1, 2])

	with col1:
		st.subheader("Analisis de Ultrasonido")

		imagen_subida = st.file_uploader("Subir Ecografía (PNG)", type=["png"])

		if imagen_subida:
			os.makedirs("temp_uploads", exist_ok=True)
			ruta_fija = "temp_uploads/current_scan.png"
			with open(ruta_fija, "wb") as f:
				f.write(imagen_subida.getbuffer())
			st.success("✅ Imagen cargada y lista para análisis.")

	with col2:
		st.subheader("Copiloto Clínico (Consulta Única)")

		chat_placeholder = st.empty()

		if prompt_medico := st.chat_input(
			"Solicita análisis de imagen, pautas de actuación..."
		):
			with chat_placeholder.container():
				with st.chat_message("user"):
					st.write(prompt_medico)

				with st.chat_message("assistant"):
					with st.spinner("Generando análisis clínico..."):
						payload = {"query": prompt_medico}

						try:
							respuesta_api = requests.post(
								f"{API_URL}/api/medico/chat", json=payload
							)

							if respuesta_api.status_code == 200:
								respuesta_texto = respuesta_api.json()["respuesta"]

								renderizar_mensaje_multimodal(respuesta_texto)

							else:
								st.error(
									f"Error en el servidor: {respuesta_api.status_code}"
								)
								st.error(f"Detalle: {respuesta_api.text}")
						except Exception as e:
							st.error(f"Error de conexión con FastAPI: {e}")

elif "pacientes" in grupos_usuario:
	st.title("Asistente de Salud Maternal")
	for msg in st.session_state.mensajes_paciente:
		with st.chat_message(msg["role"]):
			st.write(msg["content"])

	if prompt := st.chat_input("Escribe tu consulta médica aquí..."):
		st.session_state.mensajes_paciente.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.write(prompt)

		with st.chat_message("assistant"):
			with st.spinner("Analizando con seguridad..."):
				payload = {"query": prompt, "session_id": st.session_state.session_id}
				try:
					respuesta_api = requests.post(
						f"{API_URL}/api/paciente/chat", json=payload
					)
					if respuesta_api.status_code == 200:
						respuesta_texto = respuesta_api.json()["respuesta"]
						st.write(respuesta_texto)
						st.session_state.mensajes_paciente.append(
							{"role": "assistant", "content": respuesta_texto}
						)
					else:
						st.error("Error conectando con el servidor médico.")
				except Exception as e:
					st.error(f"Error de conexión: {e}")

else:
    st.error("🚨 Acceso denegado. Su usuario no tiene un rol asignado en el sistema.")
    st.stop()
