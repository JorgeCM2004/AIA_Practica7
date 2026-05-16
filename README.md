# Puesta en funcionamiento de la aplicación médica.

Este proyecto es la integración y puesta en producción de los distintos módulos realizados en prácticas anteriores (**[Práctica 1](https://github.com/JorgeCM2004/AIA_Practica1)**, **[Práctica 3](https://github.com/JorgeCM2004/AIA_Practica3)**, **[Práctica 5](https://github.com/JorgeCM2004/AIA_Practica5)**), además del uso de **[Fastapi](https://fastapi.tiangolo.com)** (Backend) y **[Streamlit](https://streamlit.io)** (Frontend) para la creación de una aplicación de carácter profesional, y su posterior encapsulación gracias al uso de Docker.

Destaca por su diseño modular orientado a objetos y el uso de **[uv](https://github.com/astral-sh/uv)** para una gestión del entorno y dependencias ultrarrápida y reproducible.

## Estructura del Proyecto

El código está organizado en distintos módulos para facilitar su lectura y mantenimiento:
```
├── 📁 utils
│   ├── 📁 agent
│   │   ├── 🐍 F_Maternal_Agent.py
│   │   ├── 🐍 F_Medical_Agent.py
│   │   └── 🐍 __init__.py
│   ├── 📁 logger
│   │   └── 🐍 F_logger.py
│   ├── 📁 searcher
│   │   ├── 🐍 F_Hybrid_Searcher.py
│   │   ├── 🐍 F_Lexical_Searcher.py
│   │   └── 🐍 F_Semantic_Searcher.py
│   ├── 📁 security
│   │   ├── 🐍 F_Encrypter.py
│   │   ├── 🐍 F_anonymizer.py
│   │   ├── 🐍 F_prompt_injection.py
│   │   └── 🐍 __init__.py
│   ├── 📁 tabular
│   │   ├── 🐍 F_Data_Preprocessor.py
│   │   └── 🐍 F_XGBoost.py
│   ├── 📁 vision
│   │   ├── 🐍 F_CNN_Classifier.py
│   │   └── 🐍 F_YOLO_Model.py
│   └── 🐍 __init__.py
├── 📁 weights
│   ├── 📄 CNN_Classifier.pth
│   ├── 📄 Data_Preprocessor.joblib
│   ├── 📄 XGBoost.joblib
│   └── 📄 YOLO.pt
├── ⚙️ .dockerignore
├── ⚙️ .gitignore
├── 📄 Dockerfile.backend
├── 📄 Dockerfile.frontend
├── 📝 README.md
├── 🐍 backend.py
├── ⚙️ docker-compose.yml
├── 🐍 frontend.py
├── ⚙️ pyproject.toml
└── 📄 uv.lock
```

## 1. Instalación de `uv`

Si aún no tienes el gestor de paquetes `uv` instalado en tu sistema, abre tu terminal y ejecuta el comando correspondiente a tu sistema operativo:

**Para macOS y Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**Para Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

## 2. Instalación de Ollama

Este proyecto puede llegar a utilizar modelos de lenguaje ejecutados en local para garantizar la privacidad de los datos. Para ello, necesitamos instalar la herramienta **Ollama** y descargar el modelo de Meta.
* **Windows y macOS:** Dirígete a la [página oficial de descarga de Ollama](https://ollama.com/download) y baja el instalador correspondiente para tu sistema.
* **Linux:** Puedes instalarlo directamente ejecutando en tu terminal:
  ```bash
  curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
  ```

## 3. Instalación de Llama3.2

Una vez instalado Ollama, asegúrate de que la aplicación está abierta. Luego, abre una nueva terminal y ejecuta el siguiente comando para descargar el cerebro de nuestro agente:
```bash
ollama run llama3.2
```
*(Nota: La primera vez que ejecutes este comando, tardará unos minutos en descargar el modelo. Una vez que termine la descarga y te aparezca un prompt de chat en la consola, puedes salir escribiendo `/bye` o cerrando la terminal).*

## 4. Configuración del Entorno

Como este proyecto utiliza `pyproject.toml` y `uv.lock`, la configuración es automática. Abre la terminal en la carpeta raíz del proyecto y ejecuta:

```bash
uv sync
```
*Este comando creará automáticamente el entorno virtual (`.venv`) e instalará las versiones exactas de las librerías (langchain, chromadb, etc.) definidas en el archivo lock, garantizando que todo funcione a la primera.*

## 5. Ejecución del Código mediante UV

Para ejecutar la aplicación debes lanzar los siguientes comandos:

Backend:
```bash
uv run backend.py
```

Frontend:
```bash
uv run frontend.py
```
Después podras acceder a la aplicación mediante: http://localhost:8501

## 6. Ejecución del Código mediante Docker
Para ejecutar la aplicación utilizando docker unicamente deberás ejecutar el comando:
```bash
docker-compose up --build -d
```
Después podras acceder a la aplicación mediante: http://127.0.0.1:8501
