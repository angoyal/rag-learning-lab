terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --------------------------------------------------------------------------
# Enable Required APIs
# --------------------------------------------------------------------------

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "eventarc.googleapis.com",
    "secretmanager.googleapis.com",
    "aiplatform.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# --------------------------------------------------------------------------
# Artifact Registry — Container Registry
# --------------------------------------------------------------------------

resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = local.name_prefix
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------
# GCS — Raw Document Storage
# --------------------------------------------------------------------------

resource "google_storage_bucket" "documents" {
  name          = "${local.name_prefix}-documents"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  public_access_prevention = "enforced"
}

# --------------------------------------------------------------------------
# Secret Manager
# --------------------------------------------------------------------------

resource "google_secret_manager_secret" "hf_token" {
  secret_id = "${local.name_prefix}-hf-token"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------
# Service Account
# --------------------------------------------------------------------------

resource "google_service_account" "app" {
  account_id   = "${local.name_prefix}-sa"
  display_name = "RAG Learning Lab service account"
}

resource "google_project_iam_member" "app_storage" {
  project = var.project_id
  role    = "roles/storage.objectUser"
  member  = "serviceAccount:${google_service_account.app.email}"
}

resource "google_project_iam_member" "app_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "app_hf_token" {
  secret_id = google_secret_manager_secret.hf_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_project_iam_member" "app_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.app.email}"
}

# --------------------------------------------------------------------------
# Cloud Run Service — Query API (always-on or scale-to-zero)
# --------------------------------------------------------------------------

locals {
  image_url = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.app.repository_id}/${var.project_name}:${var.image_tag}"
}

resource "google_cloud_run_v2_service" "query" {
  name     = "${local.name_prefix}-query"
  location = var.region

  template {
    scaling {
      min_instance_count = var.query_min_instances
      max_instance_count = var.query_max_instances
    }

    service_account = google_service_account.app.email

    containers {
      image = local.image_url
      args  = ["serve"]

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = var.query_cpu
          memory = var.query_memory
        }
      }

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.documents.name
      }

      env {
        name  = "VERTEX_MODEL_ID"
        value = var.vertex_model_id
      }

      env {
        name = "HF_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.hf_token.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# Require authentication to invoke the query API.
# Grant access only to the app service account.
# For user-facing access, put an API Gateway or IAP in front.
resource "google_cloud_run_v2_service_iam_member" "query_invoker" {
  name     = google_cloud_run_v2_service.query.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.app.email}"
}

# --------------------------------------------------------------------------
# Cloud Run Job — Ingestion (triggered by GCS upload)
# --------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "ingest" {
  name     = "${local.name_prefix}-ingest"
  location = var.region

  template {
    task_count = 1

    template {
      service_account = google_service_account.app.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = local.image_url
        args  = ["ingest", "--config", "configs/experiments/01_baseline.yaml"]

        resources {
          limits = {
            cpu    = var.ingest_cpu
            memory = var.ingest_memory
          }
        }

        env {
          name  = "GCS_BUCKET"
          value = google_storage_bucket.documents.name
        }

        env {
          name = "HF_TOKEN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.hf_token.secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------
# Cloud Run Job — Crawler (scheduled)
# --------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "crawl" {
  name     = "${local.name_prefix}-crawl"
  location = var.region

  template {
    task_count = 1

    template {
      service_account = google_service_account.app.email
      max_retries     = 1
      timeout         = "1800s"

      containers {
        image = local.image_url
        args  = ["crawl", "--query", "retrieval augmented generation", "--max-papers", "50"]

        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------
# Cloud Scheduler — Weekly Crawl
# --------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "weekly_crawl" {
  name     = "${local.name_prefix}-weekly-crawl"
  region   = var.region
  schedule = "0 6 * * 0" # every Sunday at 06:00 UTC

  http_target {
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.crawl.name}:run"
    http_method = "POST"

    oauth_token {
      service_account_email = google_service_account.app.email
    }
  }

  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------
# Eventarc — GCS Upload → Ingestion Trigger
# --------------------------------------------------------------------------

# Eventarc needs the GCS service account to publish events
data "google_storage_project_service_account" "gcs" {
}

resource "google_project_iam_member" "gcs_eventarc" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${data.google_storage_project_service_account.gcs.email_address}"
}

resource "google_project_iam_member" "eventarc_run" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.app.email}"
}

resource "google_eventarc_trigger" "ingest_on_upload" {
  name     = "${local.name_prefix}-ingest-on-upload"
  location = var.region

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }

  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.documents.name
  }

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_job.ingest.name
      region  = var.region
    }
  }

  service_account = google_service_account.app.email

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.gcs_eventarc,
  ]
}
