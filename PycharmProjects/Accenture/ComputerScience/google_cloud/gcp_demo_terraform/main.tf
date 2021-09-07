// There are some command that we would use most is:
// 1. terrarform init(init cloud provider);
// 2. terraform refresh(Tf view => real world)
// 3. terraform plan (real world => desired world)
// 4. terraform apply (desired world => real world)
// 5. terraform destroy (delete resources)

// as in the notebook to init the terraform isn't that easy, so I just start it in my own env.
// this will create some common usecase with AIA, so here just to create resources
// for windows, we have to set env variable:set GOOGLE_PROJECT=cloudtutorial-286906

// first with composer
resource "google_composer_environment" "default" {
  name = "composer-cluster-lu"
  region = "us-central1"
  config {
    node_count = 3
    node_config {
      zone = "us-central1-a"
      machine_type = "n1-standard-1"
    }
  }
}

// create bucket
resource "google_storage_bucket" "default" {
  name = "lugq_ds_test_demo_new"
  location = "US"

}

// create dataproc cluster
resource "google_dataproc_cluster" "my-cluster" {
  name = "proc-cluster-lu"
  region = "us-west1"

  cluster_config {
    master_config {
      num_instances = 1
      machine_type = "n1-standard-1"
      disk_config {
        boot_disk_size_gb = 15
      }
    }
    worker_config {
      num_instances = 2
      machine_type = "n1-standard-1"
      disk_config {
        boot_disk_size_gb = 15
      }
    }
  }
}

// bigquery dataset creation
resource "google_bigquery_dataset" "default" {
  dataset_id = "iris_dataset"
  location = "US"
  description = "Demo for iris dataset"

}

//// add with pubsub topic and subscription created for resource destroy
//resource "google_pubsub_topic" "default"{
//  name = "des-topic"
//
//}
//
//resource "google_pubsub_subscription" "default"{
//  name = "des-sub"
//  topic = google_pubsub_topic.default.name
//
//  ack_deadline_seconds = 20
//
//  // this is push
//  push_config {
//    // TODO: CHANGE THIS
//    push_endpoint = "https://example.com/push"
//
//    attributes = {
//      finish = "yes"
//    }
//  }
//}



// This is used for creating cloud function
//resource "google_storage_bucket" "bucket" {
//  name = "demo_bucket_lugq"
//}
//
//resource "google_storage_bucket_object" "archive" {
//  name   = "cloud_func.py"
//  bucket = google_storage_bucket.bucket.name
//  source = "."
//}


// create cloud function
# resource "google_cloudfunctions_function" "function" {
#   project     = "cloudtutorial-285403"
#   region      = "us-central1"
#   name        = "des-func"
#   description = "My function"
#   runtime     = "python37"

#   available_memory_mb   = 128
#   source_archive_bucket = google_storage_bucket.bucket.name
#   source_archive_object = google_storage_bucket_object.archive.name
#   entry_point           = "destroy_terra"
#   event_trigger = true
#   event_type = "providers/cloud.pubsub/eventTypes/topic.publish"
#   resource = "projects/cloudtutorial-285403/topics/des-topic"
# }

# # IAM entry for all users to invoke the function
# resource "google_cloudfunctions_function_iam_member" "invoker" {
#   project        = google_cloudfunctions_function.function.project
#   region         = google_cloudfunctions_function.function.region
#   cloud_function = google_cloudfunctions_function.function.name

#   role   = "roles/cloudfunctions.invoker"
#   member = "allUsers"
# }
