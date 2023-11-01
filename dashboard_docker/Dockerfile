# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python_docker

COPY logo.png /app/

# Déplacement des fichiers de configuration de streeamlit
# dans un répertoire .streamlit
RUN mkdir ~/.streamlit
COPY config.toml ~/.streamlit/config.toml
COPY credentials.toml ~/.streamlit/credentials.toml

EXPOSE 8501

# Lance streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["DASHBOARD.py"]