variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "rag-learning-lab"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "query_cpu" {
  description = "CPU limit for the query service (e.g. '1' or '2')"
  type        = string
  default     = "1"
}

variable "query_memory" {
  description = "Memory limit for the query service (e.g. '2Gi')"
  type        = string
  default     = "2Gi"
}

variable "ingest_cpu" {
  description = "CPU limit for the ingestion job"
  type        = string
  default     = "2"
}

variable "ingest_memory" {
  description = "Memory limit for the ingestion job"
  type        = string
  default     = "4Gi"
}

variable "query_min_instances" {
  description = "Minimum instances for the query service (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "query_max_instances" {
  description = "Maximum instances for the query service"
  type        = number
  default     = 3
}

variable "vertex_model_id" {
  description = "Vertex AI model ID for LLM generation"
  type        = string
  default     = "gemini-2.0-flash"
}
