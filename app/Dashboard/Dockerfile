FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY interface.py .

EXPOSE 8501

RUN mkdir -p .streamlit
RUN echo "\
[server]\n\
address = '0.0.0.0'\n\
port = 8501\n\
enableCORS = true\n\
" > .streamlit/config.toml

# Comando para ejecutar streamlit
CMD ["streamlit", "run", "interface.py"]