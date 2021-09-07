// this is to create a terraform composer template
// could get reference here: https://www.terraform.io/docs/providers/google/r/composer_environment.html
resource "google_composer_environment" "test" {
  name = "my-composer"
  region = "us-central1"
  // let's add config with GKE
  config {
    node_count = 3   # at least 3!
    node_config {
      zone = "us-central1-a"
      machine_type = "n1-standard-1"
      network = google_compute_network.test.id   # define cluster network
      subnetwork = google_compute_subnetwork.test.id   # this is for subnet
    }
  }
}

// network
resource "google_compute_network" "test" {
  name = "composer-network"
  auto_create_subnetworks = false
}

// subnet
resource "google_compute_subnetwork" "test" {
  name = "composer-subnet"
  ip_cidr_range = "10.2.0.0/16"
  region = "us-central1"
  network = google_compute_network.test.id
}

// service account
resource "google_service_account" "test" {
  account_id = "composer-env"
  display_name = "Test with composer terraform"
}

// iam member, we could just ignore this
//resource "google_project_iam_member" "composer-worker" {
//  member = "serviceAccount:${google_service_account.test.email}}"
//  role = "roles/composer.worker"
//}

