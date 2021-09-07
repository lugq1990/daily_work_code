// this will create some common usecase with AIA, so here just to create resources

// first with composer
resource "google_composer_environment" "test" {
  name = " proc-cluster-lu"
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
  name = "aia-terraform-lugq"
  location = "US"

}

// create dataproc cluster
resource "google_dataproc_cluster" "my-cluster" {
  name = "my-cluster"
  region = "us-central1"

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