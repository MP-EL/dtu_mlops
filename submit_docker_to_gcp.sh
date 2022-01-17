export MY_PROJECT_ID=fleet-reserve-338511
export BUCKET_NAME=mlo-bucket-test
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REGION=europe-west1
export IMAGE_URI=gcr.io/fleet-reserve-338511/testing:latest
export MODEL_DIR=pytorch_model_$(date +%Y%m%d_%H%M%S)
export JOB_NAME=custom_container_job_$(date +%Y%m%d_%H%M%S)

gcloud config set project $MY_PROJECT_ID
gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --master-image-uri $IMAGE_URI -- --model-dir=gs://$BUCKET_NAME/$MODEL_DIR
