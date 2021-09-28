### How to submit AI-platform jobs

If we want to use a `package`that stored in GCS to start our job training, then we could use bellow command to submit job:
```shell
gcloud ai-platform jobs submit training "test_ai_2" \
  --staging-bucket gs://schedule_with_platform/ \
  --module-name "iris_trainer.training" \
  --packages gs://schedule_with_platform/iris_trainer-0.1.tar.gz \
  --region us-central1 \
  --runtime-version=2.5 \
  --python-version=3.7 \
  --scale-tier basic
```

If we want to use **python** to call AI-platform job, just reference code: `cloud_function_to_trigger_ai_job.py` file.
