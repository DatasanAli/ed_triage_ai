ENDPOINT_NAME = edtriage-live

deploy-endpoint:
	python sagemaker/scripts/repack_and_deploy.py

delete-endpoint:
	aws sagemaker delete-endpoint --endpoint-name $(ENDPOINT_NAME)
	@echo "Endpoint $(ENDPOINT_NAME) deleted."
