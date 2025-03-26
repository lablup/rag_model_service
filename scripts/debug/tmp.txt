bai_manager/backendai/backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  auto_rag \
  1 \
  --name rag_reflex-web2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models/RAGModelService/rag_services/dafced48/model-definition-reflex-dev-reflex-web-docs.yaml \
  --public \
  -e RAG_SERVICE_NAME=rag_reflex-web \
  -e RAG_SERVICE_PATH=rag_services/dafced48 \
  -r mem=4g \
  -r cpu=2 \
  --bootstrap-script ./auto_rag_service/setup.sh \
  --startup-command "python3 /models/RAGModelService/auto_rag_service/start.sh"


bai_manager/backendai/backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  auto_rag \
  1 \
  --name rag_reflex-web2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --public \
  -e RAG_SERVICE_NAME=rag_reflex-web \
  -e RAG_SERVICE_PATH=rag_services/dafced48 \
  -r mem=4g \
  -r cpu=2 \
  --model-definition-path ./RAGModelService/rag_services/dafced48/model-definition-reflex-dev-reflex-web-docs.yaml