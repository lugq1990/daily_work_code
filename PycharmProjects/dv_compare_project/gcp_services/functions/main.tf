# -------------------------- PROVIDERs -------------------------- 
provider "google-beta" {
  credentials = var.gcp_credential_path
  project     = var.gcp_project_id
  region      = var.gcp_project_region
  zone        = var.gcp_project_zone
}

provider "google" {
  credentials = var.gcp_credential_path
  project     = var.gcp_project_id
  region      = var.gcp_project_region
  zone        = var.gcp_project_zone
}

terraform {
  backend "gcs" {
    bucket      = "#{TF_STATE_FILES_BUCKET}#"
    #prefix      = "terraform/#{GCP_PROJECT_ID}#/#{TOOL_TF_PIPELINE_NAME}#/#{TOOL_TF_RELEASE_ID}#"
    prefix      = "terraform/#{GCP_PROJECT_ID}#/#{Build.DefinitionName}#"
    credentials = "#{GCP_PROJECT_CREDENTIAL_FILE_PATH}#"
  }
}


# -------------------------- Project Configuration --------------------------
# Getting project data
data "google_project" "project" {
  project_id = var.gcp_project_id
}


module "bucket_storage_functions" { 
  source      = "acnciotfregistry.accenture.com/accenture-cio/storage/google"           # Module source
  version     = "#{GCP_TF_BUCKET_VERSION}#"                                                           # Module version

  # Parameters. 
  project_id     = var.gcp_project_id #Application team should provide the required value for Project ID.
  storage_class         = var.gcp_storage_class
  location              = var.gcp_project_region
  storage_name           = var.functions-code-bucket
  versioning            = var.gcp_bucket_versioning
}


module "compare_data_pubsub" {
    source = "acnciotfregistry.accenture.com/accenture-cio/pubsub/google"
    version = "#{GCP_TF_PUBSUB_VERSION}#"

  # Parameters. Configures a Pull Subscription
    project_id = var.gcp_project_id   
    topic_name = "COMPARE-DATA-TOPIC"
    intended_event_type = "Application" #or Security
    resources_to_create = "TOPIC" #or TOPIC or SUBSCRIPTION
    publisher_service_accounts  = [var.gcp_owner_service_account]
    subscriber_service_accounts = [var.gcp_owner_service_account]       
}

module "compare_data_in_lake_cf" {
  # Module source.
  source = "acnciotfregistry.accenture.com/accenture-cio/function/google"
  # Module version.
  version = "#{GCP_TF_CF_VERSION}#"

  # Parameters
  project_id                        = var.gcp_project_id
  google_function_name              = "compare-data-inlake-cloud-function"
  google_function_entrypoint        = "data_comparation"
  environment_variables            = {
    publishTopic = element(module.send_comparision_result_pubsub.topic_name,0),
    project = var.gcp_project_id
  }
  trigger_http                      = "false"
  trigger_event                     = {
    event_type = "google.pubsub.topic.publish"
    event_resource = element(module.compare_data_pubsub.topic_name,0),"retry" = false
  }
  google_storage_bucket_name        = module.bucket_storage_functions.storage_name
  path_to_data_to_upload            = "./tmp/function1.zip"
  source_dir                        = "./compare_data_v2"
  available_memory_mb               = 256 
  timeout                           = 540 
  runtime                           = "python37"
  region                            = var.gcp_project_region
  max_instances                     = var.max_instances 
  google_storage_bucket_object_name = "cloudfunctions/python/function1.zip"
  app_execution_service_account     = element(split("@",var.gcp_owner_service_account),0)
}


module "send_comparision_result_pubsub" {
    source = "acnciotfregistry.accenture.com/accenture-cio/pubsub/google"
    version = "#{GCP_TF_PUBSUB_VERSION}#"

  # Parameters. Configures a Pull Subscription
    project_id = var.gcp_project_id   
    topic_name = "SEND-COMPARISION-RESULT-TOPIC"
    subscription_name   = "dv_auto_test_sub"
    intended_event_type = "Application" #or Security
    #resources_to_create = "TOPIC" #or TOPIC or SUBSCRIPTION
    resources_to_create = "BOTH" #or TOPIC or SUBSCRIPTION
    delivery_type = "Pull" 
    publisher_service_accounts  = [var.gcp_owner_service_account]
    subscriber_service_accounts = [var.gcp_owner_service_account]       
}
