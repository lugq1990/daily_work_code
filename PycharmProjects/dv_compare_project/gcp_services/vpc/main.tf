# -------------------------- Terraform Providers -------------------------- 
#
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

# -------------------------- BACKEND --------------------------
terraform {
  backend "gcs" {
    bucket      = "#{TF_STATE_FILES_BUCKET}#"
    prefix      = "terraform/#{GCP_PROJECT_ID}#/#{Release.DefinitionName}#"
    credentials = "#{GCP_PROJECT_CREDENTIAL_FILE_PATH}#"
  }
}

# -------------------------- VPC MODULE --------------------------
module "gcp_create_vpc"{
   source       = "acnciotfregistry.accenture.com/accenture-cio/vpc/google" 
   version      = "#{GCP_TF_VPC_VERSION}#" 
   project_id   = var.gcp_project_id
   region_name  = [var.gcp_project_region]
}