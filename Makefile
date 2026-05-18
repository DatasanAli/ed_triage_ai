ENDPOINT_NAME = edtriage-live

# Requires the virtual environment to be active: source .venv/bin/activate
server:
	@echo "Reminder: make sure the virtual environment is active: source .venv/bin/activate"
	PYTHONPATH=$(CURDIR)/src:$(CURDIR) uvicorn backend.main:app --host 0.0.0.0 --port 8000

ui:
	@echo "Reminder: make sure the virtual environment is active: source .venv/bin/activate"
	streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

deploy-endpoint:
	python sagemaker/scripts/repack_and_deploy.py

delete-endpoint:
	aws sagemaker delete-endpoint --endpoint-name $(ENDPOINT_NAME)
	@echo "Endpoint $(ENDPOINT_NAME) deleted."
