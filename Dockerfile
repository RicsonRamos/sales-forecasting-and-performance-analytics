# Usa uma imagem oficial do Python otimizada
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala apenas o essencial para o Python rodar (evita erros de biblioteca)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de dependências
COPY requirements.txt .

# Instala as bibliotecas do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto para dentro do container
COPY . .

# Expõe a porta que o Streamlit usa
EXPOSE 8501

# Comando para iniciar o app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]