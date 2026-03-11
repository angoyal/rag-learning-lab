output "ecr_repository_url" {
  description = "ECR repository URL for pushing Docker images"
  value       = aws_ecr_repository.app.repository_url
}

output "alb_dns_name" {
  description = "ALB DNS name for the query service"
  value       = aws_lb.query.dns_name
}

output "s3_bucket_name" {
  description = "S3 bucket for raw documents"
  value       = aws_s3_bucket.documents.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name for running tasks"
  value       = aws_ecs_cluster.main.name
}

output "query_service_name" {
  description = "ECS service name for the query API"
  value       = aws_ecs_service.query.name
}
