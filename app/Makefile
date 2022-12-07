.PHONY: run run-container gcloud-deploy

APP_NAME ?= app

run:
	@streamlit run app.py --server.port=8501

run-container:
	@docker build . -t ${APP_NAME}
	@docker run -p 8501:8501 ${APP_NAME}

gcloud-deploy:
	@gcloud app deploy app.yaml
