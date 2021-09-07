
provider "google" {
  project = "buoyant-sum-302208"
  region  = "us-central1"
}

resource "google_pubsub_topic" "topic"{
  name = "job-topic"
}

resource "google_cloud_scheduler_job" "job"{
  name = "test-scheduler"
  description = "Just to get with terraform"
  schedule = "* * * * *"

  pubsub_target {
    topic_name = google_pubsub_topic.topic.id
    data = base64encode("Hi scheduler!")
  }
}


resource "google_storage_bucket" "bucket"{
  name = "cloud_functions_lugq"
}

# data "archive_file" "source"{
#   type = "zip"
#   source_dir = "./cloud_func_code"
#   output_path = "./cloud_func_code/index.zip"
# }

resource "google_storage_bucket_object" "archive"{
  name = "index.zip"
  bucket = google_storage_bucket.bucket.name
  source = "./cloud_func_code/index.zip"
}

resource "google_cloudfunctions_function" "function"{
  name = "func_lugq"
  description = "test with cloud function"
  runtime = "python38"

  available_memory_mb = 128
  source_archive_bucket = google_storage_bucket.bucket.name
  source_archive_object = google_storage_bucket_object.archive.name
  # trigger_http = false

  entry_point = "say_hi"

  event_trigger {
    # This have to be right!!!!! otherwise we even't don't know the failure reason.
    event_type = "google.pubsub.topic.publish"
    resource = google_pubsub_topic.topic.name

  }
}