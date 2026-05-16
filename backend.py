from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils import Hybrid_Searcher
from utils.agent import Maternal_Agent, Medical_Agent
from utils.logger.F_logger import logger

app = FastAPI(title="Maternal Health AI Server", version="1.0")
searcher = Hybrid_Searcher()
paciente_agent = Maternal_Agent(searcher=searcher)
medico_agent = Medical_Agent()


class PacienteRequest(BaseModel):
	query: str
	session_id: str


class MedicoRequest(BaseModel):
	query: str


@app.post("/api/paciente/chat")
async def chat_paciente(request: PacienteRequest):
	try:
		logger.info(f"<patient query> {request.query}")
		respuesta = paciente_agent.run(request.query)
		logger.info(f"<maternal agent response> {respuesta}")
		return {"respuesta": respuesta}
	except Exception as e:
		logger.error(f"<ERROR from patient agent> {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/medico/chat")
async def chat_medico(request: MedicoRequest):
	try:
		logger.info(f"<doctor query> {request.query}")
		respuesta = medico_agent.run(request.query)
		logger.info(f"<medical agent response> {respuesta}")
		return {"respuesta": respuesta}
	except Exception as e:
		logger.error(f"<ERROR from doctor agent> {str(e)}")
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000)
