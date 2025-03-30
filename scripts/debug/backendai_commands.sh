source bai_manager/backendai/env-dogbowl.sh

List sessions:
bai_manager/backendai/backendai-client ps


List vFolders;
bai_manager/backendai/backendai-client vfolder list

bai_manager/backendai/backendai-client/vfolder list-hosts
Default vfolder host: seoul-h100:flash01
Usable hosts: seoul-h100:flash01, seoul-h100:flash02, seoul-h100:flash03


Create new vFolder:
bai_manager/backendai/backendai-client/backendai-client vfolder create reflexdev_rag_service seoul-h100:flash03 --usage-mode "model"


bai_manager/backendai/backendai-client create --owner-access-key AKIATRQHTZ2Z4IJ7ICAB \
  --startup-command "echo /home/work/my-vol/script.py" \
  --bootstrap-script ./setup_test.sh \
  --tag "rag_deploy_session" \
  --architecture x86_64 \
  --type interactive \
  --name reflexdev_rag_service_session \
  --env VAR1=value1 \
  --env VAR2=value2 \
  --volume reflexdev_rag_service=my-vol \
  --resources cpu=4 --resources mem=8g --resources cuda.shares=1 \
  --group default \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2

Session ID 35203741-6a7f-4ccb-b2ea-47e6d9c65422 is created and ready.
∙ This session provides the following app services: sshd, ttyd, jupyter, jupyterlab, vscode, tensorboard, mlflow-ui, nniboard


SSH into B.AI Remote Session, interactive:
bai_manager/backendai/backendai-client session ssh 35203741-6a7f-4ccb-b2ea-47e6d9c65422

Copy Current project into remote directory. vFolder:
backend.ai session scp YOUR_SESSION_NAME -p 9922 -r ./ work@localhost:/home/work/my-vol/

backend.ai service create cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 08fc5b55-370a-4793-b582-f167309a6f0 1 -r gpu=5 -r mem=32 -r cpu=4 --tag agentic --name agentic

# full_version
./backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  ai_apps \
  1 \
  --name readlm \
  --tag rag_model_service \
  --project default \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --model-definition-path model-definition-readlm.yaml \
  --mount reflexdev_rag_service \
  --public \
  -o AKIATRQHTZ2Z4IJ7ICAB \
  -r cuda.shares=4 \
  -r mem=32g \
  -r cpu=4
  
  # short version
  ./backendai-client service create \
  cr.backend.ai/cloud/ngc-pytorch:23.09-pytorch2.1-py310-cuda12.2 \
  ai_apps \
  1 \
  --name readlm2 \
  --tag rag_model_service \
  --scaling-group nvidia-H100 \
  --model-mount-destination /models \
  --model-definition-path model-definition-readlm.yaml \
  --public \
  -r cuda.shares=4 \
  -r mem=32g \
  -r cpu=4
  
  
  