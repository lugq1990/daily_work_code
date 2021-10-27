variable "gcp_credential_path" {
  type        = string
  description = "the gcp credential path"
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
  default     = ""
}
