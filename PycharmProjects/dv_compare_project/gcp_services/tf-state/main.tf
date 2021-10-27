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

# -------------------------- Bucket creation -------------------------- 
module "gcp_storage" {
  source        = "acnciotfregistry.accenture.com/accenture-cio/storage/google"
  version       = "#{GCP_TF_BUCKET_VERSION}#"
  project_id    = var.gcp_project_id
  storage_class = "REGIONAL"
  location      = var.gcp_project_region
  storage_name  = "tf-state"
  versioning    = true
  labels        = {aiaid        : "#{APP_NUMBER}#",
                   application  : "#{APP_NAME}#",
                   created-via  : "devops-release-pipeline"}
}
