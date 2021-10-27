variable "gcp_credential_path" {
  type        = string
  description = "the gcp credential path"
}

variable "gcp_owner_service_account" {
  description = "The project sa with owner privileges"
  type        = string
  default     = ""
}

variable "gcp_app_service_account" {
  description = "The project sa with app privileges"
  type        = string
  default     = ""
}

variable "gcp_bd_service_account" {
  description = "BD service account with storage.bucket.legacy.owner privilege in the buckets sited on the lake"
  type        = string
  default     = ""
}

variable "gcp_project_id" {
  type        = string
  description = "the internal project id of your project "
}

variable "gcp_project_region" {
  type        = string
  description = "project region of your project"
  default     = "us-east1"
}

variable "gcp_project_zone" {
  type        = string
  description = "project zone of your project"
  default     = "us-east1-a"
}

variable "tf_release_id" {
  type        = string
  description = "the unique id for the Azure DevOps release id used to store terraform state in GCP"
}

variable "gcp_storage_class" {
  description = "GCP Storage bucket class :  Content must be either MULTI_REGIONAL, REGIONAL, NEARLINE or COLDLINE"
  type        = string
  default     = "REGIONAL"
}

variable "gcp_storage_type" {
  description = "Defines the scope of accessibility for the storage bucket, if it is publicly accessible (public) or just internal to Accenture infrastructure (private)." # Content must be either "private" or "public". Default is "private"  type        = string
  default     = "private"
}

variable "functions-code-bucket" {
  description = "Bucket that contains Cloud Functions Code"
  type        = string
  default = "cloud-functions-code-1024"
}

variable "gcp_bucket_versioning" {
  description = "Enable bucket Level versioning feature to protect the Cloud Storage data from being overwritten or accidentally deleted."
  type        = string
  default     = "false"
}

variable "max_instances" {
  description = "Max instances in cloud functions."
  type        = string
}
