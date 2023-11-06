# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python_docker

# Répertoire de travail
WORKDIR /app

COPY API.py /app/
COPY xgb.json /app/
COPY application_test.zip /app/
COPY application_train.zip /app/

# Déclaration des ports d'entrées à l'app depuis l'extérieur du container
EXPOSE 5001

# Lance Flask
CMD ["python", "API.py"]