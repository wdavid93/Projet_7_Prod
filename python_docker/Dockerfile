FROM python:3.9-slim

# Répertoire de travail
RUN mkdir /app
WORKDIR /app

# Ajouts de la liste des bibliothèques python
COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
