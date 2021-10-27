# Introduction
The purpose of this Terraform program *'tf-code/tf.gcs.bucket.support.tf.state'* within Repo *'65343-aia-tf-v2-template'* is to **create the GCS Bucket that will support Terraform State for all your GCP Project Pipelines**.

> [!IMPORTANT]
> **WARNING!!!: This Terraform Template Pipeline should be the first Terraform pipeline that you deploy in your GCP project**, before any other pipeline, like the one to configure VPC is deployed. This is due that any other pipeline must leverage the GCS Bucket, which is created with this pipeline, to store Terraform state files of your pipelines.  
> 
Terraform must store state about your managed infrastructure and configuration. This state is used by Terraform to map real world resources to your configuration, keep track of metadata, and to improve performance for large infrastructures.
This state is stored by default in a file with extension *'.tfstate'*, in JSON format. Terraform uses this state to create plans and make changes to your infrastructure.   
  
All remaining GCP AIA sample pipelines are configured to use the following consistent Terraform Back-End State Configuration to store Terraform state using a GCS bucket where the state is saved. Hence, it is required that you deploy this *'aia-gcs-bucket-support-tf-state-22-template'* before any other pipeline in your GCP Processing project. Otherwise, your pipelines will fail.
The bucket below with value TF_STATE_FILES_BUCKET is the GCP bucket that this Terraform program will create. TF_STATE_FILES_BUCKET wil take the value 'your-GCP-project-name' + '-tf-state' (e.g. 'npd-65343-dv2aiaarch-bd-tf-state'). 

```ruby
# https://www.terraform.io/docs/backends/types/gcs.html
 terraform {
   backend "gcs" {
     bucket      = "#{TF_STATE_FILES_BUCKET}#"
     prefix      = "terraform/#{GCP_PROJECT_ID}#/#{TOOL_TF_PIPELINE_NAME}#"
     credentials = "#{TF_STATE_FILES_CREDENTIALS}#"
   }
 }
```
In this way, your pipelines will be saving its Terraform state file in the TF_STATE_FILES_BUCKET bucket in your Processing project, at the path "terraform/#{GCP_PROJECT_ID}#/#{TOOL_TF_PIPELINE_NAME}#". TF_STATE_FILES_CREDENTIALS is the credentials key file of your IAC deployment service account with owner rights in your GCP project.
See a sample picture below of the TF state file created using the Composer pipeline
![Image-tf-state](images/tf-state-img-1.PNG)

# Pipeline
The AzDevOps pipeline that will deploy this Terraform program can be located within this repository at the path *'pipelines-templates/aia-gcs-bucket-support-tf-state-22-template.json'*. It is in JSON format so that you can import it easily into your AzDevOps project.   
The pipeline contains one artefact that is a git repository (*65343-aia-tf-v2-template*). This git will be downloaded into the temporal machine (agent server). The Pipeline will deploy the Terraform program located at path */tf-code/tf.gcs.bucket.support.tf.state*. 


## GCP Credentials file – Secure Files
Pipeline will have to use a GCP Credentials JSON key file to authenticate to the GCP project when running gcloud commands and the GCS Bucket Support TF State Terraform script. This key is for the GCP Service Account with owner rights that was created as part of the GCP project provisioning process in CAPP tool.
To secure this GCP Credentials JSON key, it is uploaded to the Secure Files Library in Azure DevOps Pipelines so that contents of the secure files are encrypted and can only be used during the release pipeline by referencing them from a task.   


## Variables
This pipeline use only local pipeline variables. Variables groups are not used to simplify exporting/importing the pipeline release across DevOps projects.
Following is a description of the pipelines variables

### Not settable at release time
* GCP_BUCKET_VERSIONING: Whether the GCS Bucket TF State will allow versioning. By default set to true.
* GCP_PROJECT_CREDENTIAL_FILE_PATH: Agent temporary folder where GCP credentials secure file is downloaded after running a Download Secure File task. When the pipeline job completes, no matter whether it succeeds, fails, or is canceled, the secure file is deleted.  
* GCP_PROJECT_REGION:  Project region of your GCP project where GCS Bucket TF State will be deployed. (e.g 'us-east1')  
* GCP_PROJECT_ZONE: Project region of your GCP project where GCS Bucket TF State will be deployed. (e.g 'us-east1-d')    
* GCP_STORAGE_CLASS: Storage class of the GCS Bucket TF State. By default set to 'REGIONAL'
* TOOL_PATH: Path in the pipeline agent server where software will be installed. By default set to '$(System.DefaultWorkingDirectory)/tools'
* TOOL_TF_DOWNLOAD_URL: URL to Hashicorp Terraform site where Terraform tool for Linux is available to download. By default set to 'https://releases.hashicorp.com/terraform/$(TOOL_TF_VERSION)/terraform_$(TOOL_TF_VERSION)_linux_amd64.zip' 
* TOOL_TF_PATH: Path in the pipeline agent server where Terraform tool is installed. By default set to '$(System.DefaultWorkingDirectory)/tools'
* TOOL_TF_PIPELINE_NAME: The name of the release pipeline to which the current release belongs ('$(Release.DefinitionName)'). It is used to build the path where Terraform state files for this pipeline will be saved.  
* TOOL_TF_RELEASE_ID: The name and identifier of the current release ('$(Release.ReleaseId)--$(Release.ReleaseName)'). It is used to build the path where Terraform state files for this pipeline will be saved.   
* TOOL_TF_VERSION: Terraform version to download and install to deploy the GCS Bucket TF State. (e.g.'0.12.24')  
* WORKING_DIRECTORY: Path to the working directory to run Terraform in the agent server. $(System.DefaultWorkingDirectory)/$(GIT_ARTIFACT_SOURCE_ALIAS)$(TF_CODE_PATH)

### Settable at release time
* GCP_ADDTNL_LABELS: additional custom labels that you can add to your bucket as key-value pairs (e.g {"aiaid": "your-aia-application-number-here","application":"your-application-name-here","created-via":"devops-release-pipeline"})
* GCP_PROJECT_ID: The project id where the GCS Bucket TF State will be created (e.g. 'prd-65343-datalake-bd-88394358')
* GIT_ARTIFACT_SOURCE_ALIAS: The source alias value defined in the pipeline for the GIT repository artifact (e.g. '_65343-aia-tf-v2-template')
* TF_CODE_PATH: Path within the GIT Repo to the Terraform program that the Pipeline Release will deploy (e.g. '/tf-code/tf.gcs.bucket.support.tf.state')

All the values we want to be replaced in our scripts or codes should be instanced here.


## Stages
This pipeline has only one stage __GCS Bucket Deployment__ . Its Deployment is setup to be triggered just after release. Note that there is no Destroy stage, since Terraform destroy command is based on the existence of state and this pipeline is actually used to create the GCS bucket which will be used to store stage for any other pipeline. Hence destroy cannot be used for this pipeline. 

At this time we are using as machine for our pipelines __Azure Pipelines Ubuntu 16.04__ maybe in a future this will be changed for an image managed by *Enterprise Architecture*.

Let's see the steps for each stage:
### GCS Bucket Deployment
#### 1. AzureDevOps Build Agent TF Cli Config File Generator
This allows our pipeline to connect to CCS Repository to take the Terraform buckets.

#### 2. GCP - Setup Secure Infra Credentials
This step will download the necessary GCP credentials to deploy infrastructure in your GCP project using Terraform.

#### 3. Terraform - Set-Up - Replace Tokens
On this step we replace some variables in one file:
```
terraform.tfvars=>terraform.tfvars
```
This will take the file tokenized.terraform._tfvars, copy it a new name (terraform.tfvars) and will replace the variables in the file.
The values to use when replacing the variable are those defined on the variables section of the pipeline. Remember that they should be written between `#{` and `}#`

#### 4. Terraform - Setup & Apply
On this step, basically we issue the Terraform commands  __Init__, __Plan__ and __Apply__ of our terraform code/configuration.

#### 5. Clean Agent Directories
This step deletes all the files downloaded into the virtual machine. Nothing to configure here.



# Terraform Code
There are 4 terraform files:

* main.tf: very simple program, it simply uses *'acnciotfregistry.accenture.com/accenture-cio/storage/google'* CCS Terraform module, version *'0.1.0-beta6'* to create the GCS Bucket that will be used to store Terraform State for all remaining AzDevOps pipelines in this GCP project. 
* outputs.tf: file where the output variables of the process are defined.
* terraform.tfvars: File where the process variables are initializated. 
* variables.tf: File where all the input variables used in this process are defined.  


## Overview
This Terraform configuration/template creates the GCS Bucket that will be used to store Terraform State in this GCP project. It leverage the [`CCS Google Cloud Platform Storage Module`] https://ciotfregistry.accenture.com/modules/accenture-cio/google/storage/v0_1_0-beta6/readme/ to create the GCS Bucket based on Accenture CIO Standard. See documentation in the aforemention link for details about it.


## Resources created 
A GCP GCS Bucket is created which will be referenced by any other AzDevOps Terraform Pipeline in the project to keep Terraform State files. 

## Prerequisites
- Terraform application (this script built & tested with version 0.12.24)
- Terraform **`google`** provider (this script built & tested with google provider version "3.22.0")
- Existing Google Cloud Project-ID
- Existing service account JSON key file running this Terraform script with enough rights to create GCS Bucket in the GCP Project (Custom Role Admin). The service account provided to Processing projects with owner rights is the intented service account to use.

## Usage 
This Terraform snippet creates a GCS bucket using as bucket name '{GCP_PROJECT_NAME}' + '-tf-state'. This Terraform template is already ready as it is provided to be executed. You will only need to provide your own values for the setteable pipeline variables (GCP_ADDTNL_LABELS, GCP_PROJECT_ID, GIT_ARTIFACT_SOURCE_ALIAS and TF_CODE_PATH) described above, when creating a Pipeline release to deploy the GCS Bucket.

```ruby
#  https://ciotfregistry.accenture.com/modules/accenture-cio/google/storage/v0_1_0-beta6/readme/
module "gcp_storage" { 
  
  source      = "acnciotfregistry.accenture.com/accenture-cio/storage/google"           # Module source
  version     = "0.1.0-beta6"                                                          # Module version

  # Parameters. 
    # project_id = “ ID of the project where bucket is going to create- Interpolate the output from project module”
    project_id                = var.gcp_project_id #Application team should provide the required value for Project ID.
    # storage_class = “ Application team should provide the required value for bucket class “
    # Content must be either "MULTI_REGIONAL", "REGIONAL", "NEARLINE" or "COLDLINE"
    storage_class             = var.gcp_storage_class
    # location = “ Application team should provide the required bucket location “
    location                  = var.gcp_project_region
    # storage_name = User-defined name of the resource that will be used in combination with CIO Naming convention.
    storage_name              = "tf-state"
    # Enable bucket Level versioning feature to protect the Cloud Storage data from being overwritten or accidentally deleted.
    # Content should be "true" or "false"
    versioning                = var.gcp_bucket_versioning
}
```

## Input
| Name | Description | Type | Default | Required | Example |
|------|-------------|-----|---------|----------|---------|
| gcp_credential_path | Path to the crediantials key file of the service account to be used by Terraform to run the script. | String | - | Yes | '$(credential.secureFilePath)' |
| gcp_project_id | The project id where the GCS Bucket TF State will be created. | String | - | Yes | 'npd-65343-dv2aiaarch--cc680837' |
| gcp_project_region | The project region of your project. | String | - | Yes | 'us-east1' |
| gcp_project_zone | The project zone of your project. | String | - | Yes | 'us-east1-d' |
| tf_release_id | The unique id for the Azure DevOps pipeline release id and version used to store terraform state in GCP | String | - | Yes | '12044--Release-3' |
| gcp_storage_class | GCP Storage bucket class :  Content must be either MULTI_REGIONAL, REGIONAL, NEARLINE or COLDLINE. | String | 'REGIONAL' | No | 'REGIONAL' |
| gcp_bucket_versioning | Enable bucket Level versioning feature to protect the Cloud Storage data from being overwritten or accidentally deleted. | String | 'false' | No | 'true' |
| gcp_addtnl_labels | Map of additional key/value pairs that are assigned to the gcs bucket as custom labels. Labels 'airid', 'environment' and 'tf_module_label' are already added automatically and do not need to be considered". | Map | {} | No | '{"aia_project_id": "0003", "aia_project_name":"alice","data_owner":"john_doe", "point_of_contact": "jane_doe"}' |
| gcp_notification_topic_name | Existing Pub/Sub Topic where notifications related to the bucket will be written. If left blank, notifications will not be configured. | string | '' | No |  'App_65343_mrdr_rw-1033-processingevents' |
| gcp_notification_event_types | Event types to notify. [OBJECT_FINALIZE, OBJECT_METADATA_UPDATE, OBJECT_DELETE, OBJECT_ARCHIVE]. | list | ["OBJECT_FINALIZE", "OBJECT_METADATA_UPDATE", "OBJECT_DELETE", "OBJECT_ARCHIVE"] | No |  ["OBJECT_FINALIZE"] |

## Output
| Name | Description |
|------|-------------|
| gcs_bucket_name | Storage Bucket name created|
| gcs_bucket_url | Link URL of the Storage Bucket |
| tf_module_name | Terraform module name. |
| tf_module_version | Terraform module version |

