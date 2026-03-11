output "artifact_registry_url" {
  description = "Artifact Registry URL for pushing Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.app.repository_id}"
}

output "query_service_url" {
  description = "Cloud Run URL for the query service"
  value       = google_cloud_run_v2_service.query.uri
}

output "gcs_bucket_name" {
  description = "GCS bucket for raw documents"
  value       = google_storage_bucket.documents.name
}

output "ingest_job_name" {
  description = "Cloud Run Job name for manually triggering ingestion"
  value       = google_cloud_run_v2_job.ingest.name
}

output "crawl_job_name" {
  description = "Cloud Run Job name for manually triggering crawling"
  value       = google_cloud_run_v2_job.crawl.name
}
