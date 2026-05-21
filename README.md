# Puesta en funcionamiento de la aplicación médica.

Este proyecto es la finalización y mejora de los distintos módulos realizados en prácticas anteriores.

## Estructura del Proyecto

El código está organizado en distintos módulos para facilitar su lectura y mantenimiento:
```
├── 📁 data
│   └── 📄 ghana_maternal_health.csv
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
│   │   ├── 🐍 F_Explainer.py
│   │   └── 🐍 F_XGBoost.py
│   ├── 📁 vision
│   │   ├── 🐍 F_CNN_Classifier.py
│   │   └── 🐍 F_UNet_Model.py
│   └── 🐍 __init__.py
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

## Entrar en la aplicación.
En esta práctica se ha conseguido en despliegue en cloud de la aplicación por lo que para probarla solo debes entrar en **<https://practica7.cheesyrat.com>** y entrar con un usuario permitido.
