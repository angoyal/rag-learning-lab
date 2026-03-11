variable "project_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "rag-learning-lab"
}

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
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
  description = "CPU units for the query service (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "query_memory" {
  description = "Memory (MiB) for the query service"
  type        = number
  default     = 2048
}

variable "ingest_cpu" {
  description = "CPU units for the ingestion task"
  type        = number
  default     = 2048
}

variable "ingest_memory" {
  description = "Memory (MiB) for the ingestion task"
  type        = number
  default     = 4096
}

variable "query_desired_count" {
  description = "Number of query service instances"
  type        = number
  default     = 1
}

variable "bedrock_model_id" {
  description = "Bedrock model ID for LLM generation"
  type        = string
  default     = "anthropic.claude-sonnet-4-6-20250514"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS on the ALB"
  type        = string
}
