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
    prefix      = "terraform/#{GCP_PROJECT_ID}#/#{TOOL_TF_PIPELINE_NAME}#/#{TOOL_TF_RELEASE_ID}#"
    credentials = "#{GCP_PROJECT_CREDENTIAL_FILE_PATH}#"
  }
}

# -------------------------- Bucket creation -------------------------- 
module "gcp_ai_job_#{TOOL_TF_RELEASE_ID}#" {
  source = "acnciotfregistry.accenture.com/accenture-cio/aiplatformjob/google"
  version = "1.0.1"

  project_id = var.gcp_project_id
}
