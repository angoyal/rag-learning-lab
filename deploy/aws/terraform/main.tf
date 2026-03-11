terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${local.name_prefix}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${local.name_prefix}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "${local.name_prefix}-private-${count.index}" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-igw" }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${local.name_prefix}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${local.name_prefix}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-public-rt" }

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-private-rt" }

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# --------------------------------------------------------------------------
# ECR — Container Registry
# --------------------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = local.name_prefix
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --------------------------------------------------------------------------
# S3 — Raw Document Storage
# --------------------------------------------------------------------------

resource "aws_s3_bucket" "documents" {
  bucket = "${local.name_prefix}-documents"
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --------------------------------------------------------------------------
# ECS Cluster
# --------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = local.name_prefix

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# --------------------------------------------------------------------------
# IAM — Task Execution & Task Roles
# --------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name_prefix}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "${local.name_prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.bedrock_model_id}"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
        ]
        Resource = aws_secretsmanager_secret.hf_token.arn
      },
    ]
  })
}

# --------------------------------------------------------------------------
# Secrets Manager
# --------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "hf_token" {
  name                    = "${local.name_prefix}/hf-token"
  recovery_window_in_days = 7
}

# --------------------------------------------------------------------------
# CloudWatch Log Groups
# --------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "query" {
  name              = "/ecs/${local.name_prefix}/query"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "ingest" {
  name              = "/ecs/${local.name_prefix}/ingest"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "crawl" {
  name              = "/ecs/${local.name_prefix}/crawl"
  retention_in_days = 30
}

# --------------------------------------------------------------------------
# Security Groups
# --------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name   = "${local.name_prefix}-alb"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs" {
  name   = "${local.name_prefix}-ecs"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --------------------------------------------------------------------------
# ALB — Application Load Balancer for Query Service
# --------------------------------------------------------------------------

resource "aws_lb" "query" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "query" {
  name        = "${local.name_prefix}-query"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.query.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "query" {
  load_balancer_arn = aws_lb.query.arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.query.arn
  }
}

# --------------------------------------------------------------------------
# ECS Task Definitions
# --------------------------------------------------------------------------

resource "aws_ecs_task_definition" "query" {
  family                   = "${local.name_prefix}-query"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.query_cpu
  memory                   = var.query_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "query"
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    command   = ["serve"]
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "S3_BUCKET", value = aws_s3_bucket.documents.id },
      { name = "BEDROCK_MODEL_ID", value = var.bedrock_model_id },
    ]

    secrets = [{
      name      = "HF_TOKEN"
      valueFrom = aws_secretsmanager_secret.hf_token.arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.query.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "query"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "ingest" {
  family                   = "${local.name_prefix}-ingest"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ingest_cpu
  memory                   = var.ingest_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "ingest"
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    command   = ["ingest", "--config", "configs/experiments/01_baseline.yaml"]
    essential = true

    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "S3_BUCKET", value = aws_s3_bucket.documents.id },
    ]

    secrets = [{
      name      = "HF_TOKEN"
      valueFrom = aws_secretsmanager_secret.hf_token.arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ingest.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ingest"
      }
    }
  }])
}

resource "aws_ecs_task_definition" "crawl" {
  family                   = "${local.name_prefix}-crawl"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "crawl"
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    command   = ["crawl", "--query", "retrieval augmented generation", "--max-papers", "50"]
    essential = true

    environment = [
      { name = "S3_BUCKET", value = aws_s3_bucket.documents.id },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.crawl.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "crawl"
      }
    }
  }])
}

# --------------------------------------------------------------------------
# ECS Service — Query (always-on)
# --------------------------------------------------------------------------

resource "aws_ecs_service" "query" {
  name            = "${local.name_prefix}-query"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.query.arn
  desired_count   = var.query_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.ecs.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.query.arn
    container_name   = "query"
    container_port   = 8000
  }
}

# --------------------------------------------------------------------------
# EventBridge — Scheduled Crawl (weekly)
# --------------------------------------------------------------------------

resource "aws_iam_role" "eventbridge_ecs" {
  name = "${local.name_prefix}-eventbridge-ecs"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "eventbridge_ecs" {
  name = "${local.name_prefix}-eventbridge-run-task"
  role = aws_iam_role.eventbridge_ecs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = aws_ecs_task_definition.crawl.arn
      },
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.ecs_task.arn,
        ]
      },
    ]
  })
}

resource "aws_cloudwatch_event_rule" "weekly_crawl" {
  name                = "${local.name_prefix}-weekly-crawl"
  schedule_expression = "rate(7 days)"
}

resource "aws_cloudwatch_event_target" "crawl" {
  rule     = aws_cloudwatch_event_rule.weekly_crawl.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.eventbridge_ecs.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.crawl.arn
    launch_type         = "FARGATE"
    task_count          = 1

    network_configuration {
      subnets         = aws_subnet.private[*].id
      security_groups = [aws_security_group.ecs.id]
    }
  }
}

# --------------------------------------------------------------------------
# S3 Event → Ingestion Trigger
# --------------------------------------------------------------------------

resource "aws_s3_bucket_notification" "ingest_trigger" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingest_trigger.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "raw/"
    filter_suffix       = ".pdf"
  }
}

resource "aws_iam_role" "lambda_ingest_trigger" {
  name = "${local.name_prefix}-lambda-ingest-trigger"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_ingest_trigger" {
  name = "${local.name_prefix}-lambda-trigger-policy"
  role = aws_iam_role.lambda_ingest_trigger.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = aws_ecs_task_definition.ingest.arn
      },
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.ecs_task.arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/aws/lambda/${local.name_prefix}-ingest-trigger:*"
      },
    ]
  })
}

resource "aws_lambda_function" "ingest_trigger" {
  function_name = "${local.name_prefix}-ingest-trigger"
  runtime       = "python3.11"
  handler       = "index.handler"
  role          = aws_iam_role.lambda_ingest_trigger.arn
  timeout       = 30

  filename         = data.archive_file.lambda_trigger.output_path
  source_code_hash = data.archive_file.lambda_trigger.output_base64sha256

  environment {
    variables = {
      ECS_CLUSTER     = aws_ecs_cluster.main.name
      TASK_DEFINITION = aws_ecs_task_definition.ingest.arn
      SUBNETS         = join(",", aws_subnet.private[*].id)
      SECURITY_GROUPS = aws_security_group.ecs.id
    }
  }
}

resource "aws_lambda_permission" "s3_invoke" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingest_trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.documents.arn
}

data "archive_file" "lambda_trigger" {
  type        = "zip"
  output_path = "${path.module}/lambda_trigger.zip"

  source {
    content  = <<-PYTHON
import os, json, boto3

ecs = boto3.client("ecs")

def handler(event, context):
    ecs.run_task(
        cluster=os.environ["ECS_CLUSTER"],
        taskDefinition=os.environ["TASK_DEFINITION"],
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": os.environ["SUBNETS"].split(","),
                "securityGroups": [os.environ["SECURITY_GROUPS"]],
            }
        },
    )
    return {"statusCode": 200}
PYTHON
    filename = "index.py"
  }
}
