

echo "Submitting BIGOT to the cloud"

BUCKET_NAME=bigot-trials

IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-cpu.1-4

JOB_NAME=bigot_job_$(date +%Y%m%d_%H%M%S)

PACKAGE_PATH=./trainer

REGION=us-east1

JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

TARGET_PATH=gs://sent-embeddings

SOURCE_PATH=gs://triple_vectors

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC \
    --job-dir ${JOB_DIR} \
    --module-name trainer.task \
    --package-path ${PACKAGE_PATH} \
    -- \
    --source-path ${SOURCE_PATH}\
    --target-path ${TARGET_PATH} \
    --dataset nytfb

gcloud ai-platform jobs stream-logs ${JOB_NAME}
