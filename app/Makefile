.PHONY: run run-container gcloud-deploy

APP_NAME ?= app

run:
	@streamlit run app.py --server.port=8585 --server.address=0.0.0.0

run-container:
	@docker build . -t ${APP_NAME}
	@docker run -p 8585:8585 ${APP_NAME}

gcloud-deploy:
	@gcloud app deploy app.yaml
