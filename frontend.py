import json
import os
import re
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="AIA_P7", layout="wide")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

headers = st.context.headers if hasattr(st.context, "headers") else {}
grupos_usuario = headers.get("X-authentik-groups", "doctores").lower()

nombre_usuario = headers.get("X-authentik-username", "usuario_local").lower()

CARPETA_MEMORIAS = "messages"
os.makedirs(CARPETA_MEMORIAS, exist_ok=True)

ARCHIVO_MEMORIA = os.path.join(CARPETA_MEMORIAS, f"memoria_{nombre_usuario}.json")


def cargar_memoria():
	if os.path.exists(ARCHIVO_MEMORIA):
		try:
			with open(ARCHIVO_MEMORIA, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception as e:
			print(e)
			return None
	return None


def guardar_memoria():
	with open(ARCHIVO_MEMORIA, "w", encoding="utf-8") as f:
		json.dump(st.session_state.historiales_chat, f, ensure_ascii=False, indent=4)


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


if "doctores" in grupos_usuario:
	st.title("Dashboard Clínico")

	col1, col2 = st.columns([1, 2])
	with st.sidebar:
		st.link_button(
			"🚪 Cerrar Sesión",
			"/outpost.goauthentik.io/sign_out",
			use_container_width=True,
		)

	with col1:
		st.subheader("Analisis de Ultrasonido")

		imagenes_subidas = st.file_uploader(
			"Subir Ecografías (PNG)",
			type=["png"],
			accept_multiple_files=True,
		)

		if imagenes_subidas:
			os.makedirs("temp_uploads", exist_ok=True)
			archivos_guardados = []
			for idx, img in enumerate(imagenes_subidas):
				nombre_seguro = f"{idx}_{img.name.replace(' ', '_')}"
				ruta = os.path.join("temp_uploads", nombre_seguro)
				with open(ruta, "wb") as f:
					f.write(img.getbuffer())
				archivos_guardados.append(nombre_seguro)
			st.session_state["uploaded_files"] = archivos_guardados
			n = len(archivos_guardados)
			st.success(f"✅ {n} imagen{'es' if n != 1 else ''} lista{'s' if n != 1 else ''} — el copiloto las analizará cuando lo solicites.")
		else:
			st.session_state.setdefault("uploaded_files", [])

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
						payload = {
							"query": prompt_medico,
							"uploaded_files": st.session_state.get("uploaded_files", []),
						}

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
	if "historiales_chat" not in st.session_state:
		memoria_guardada = cargar_memoria()

		if memoria_guardada:
			st.session_state.historiales_chat = memoria_guardada
			st.session_state.session_id_actual = list(memoria_guardada.keys())[-1]
		else:
			default_id = str(uuid.uuid4())
			st.session_state.historiales_chat = {default_id: []}
			st.session_state.session_id_actual = default_id

	with st.sidebar:
		st.link_button(
			"🚪 Cerrar Sesión",
			"/outpost.goauthentik.io/sign_out",
			use_container_width=True,
		)

		st.title("💬 Tus Chats")

		if st.button("➕ Nuevo Chat", use_container_width=True):
			nuevo_id = str(uuid.uuid4())
			st.session_state.historiales_chat[nuevo_id] = []
			st.session_state.session_id_actual = nuevo_id
			guardar_memoria()
			st.rerun()

		st.divider()

		st.subheader("Historial")
		for s_id, mensajes in st.session_state.historiales_chat.items():
			titulo = (
				mensajes[0]["content"][:20] + "..." if mensajes else "Nuevo Chat vacío"
			)

			if s_id == st.session_state.session_id_actual:
				titulo = f"👉 {titulo}"

			if st.button(titulo, key=s_id, use_container_width=True):
				st.session_state.session_id_actual = s_id
				st.rerun()

		st.divider()

		st.subheader("Opciones")

		if st.button(
			"🗑️ Borrar chat actual", use_container_width=True, type="secondary"
		):
			id_borrar = st.session_state.session_id_actual
			del st.session_state.historiales_chat[id_borrar]

			if not st.session_state.historiales_chat:
				nuevo_id = str(uuid.uuid4())
				st.session_state.historiales_chat[nuevo_id] = []
				st.session_state.session_id_actual = nuevo_id
			else:
				st.session_state.session_id_actual = list(
					st.session_state.historiales_chat.keys()
				)[-1]

			guardar_memoria()
			st.rerun()

	st.title("Asistente de Salud Maternal")
	id_actual = st.session_state.session_id_actual

	for msg in st.session_state.historiales_chat[id_actual]:
		with st.chat_message(msg["role"]):
			st.write(msg["content"])

	if prompt := st.chat_input("Escribe tu consulta médica aquí..."):
		st.session_state.historiales_chat[id_actual].append(
			{"role": "user", "content": prompt}
		)
		guardar_memoria()

		with st.chat_message("user"):
			st.write(prompt)

		with st.chat_message("assistant"):
			with st.spinner("Analizando con seguridad..."):
				payload = {"query": prompt, "session_id": id_actual}

				try:
					respuesta_api = requests.post(
						f"{API_URL}/api/paciente/chat", json=payload
					)
					if respuesta_api.status_code == 200:
						respuesta_texto = respuesta_api.json()["respuesta"]
						st.write(respuesta_texto)

						st.session_state.historiales_chat[id_actual].append(
							{"role": "assistant", "content": respuesta_texto}
						)
						guardar_memoria()
				except Exception as e:
					st.error(f"Error de conexión: {e}")

elif "admin" in grupos_usuario:
	with st.sidebar:
		st.link_button(
			"🚪 Cerrar Sesión",
			"/outpost.goauthentik.io/sign_out",
			use_container_width=True,
		)

	st.title("Panel de control")

else:
	st.error("🚨 Acceso denegado. Su usuario no tiene un rol asignado en el sistema.")
	st.link_button(
		"Back",
		"/outpost.goauthentik.io/sign_out",
		width="content",
	)
	st.stop()
