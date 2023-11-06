az login
az acr login --name monappregistrywd

az acr build --registry monappregistrywd --resource-group appgroup --image python_docker ./python_docker
az acr build --registry monappregistrywd --resource-group appgroup --image api_docker ./api_docker
az acr build --registry monappregistrywd --resource-group appgroup --image dashboard_docker ./dashboard_docker

docker context create aci acicontextwd
docker context use acicontextwd
docker compose up
