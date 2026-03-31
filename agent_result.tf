# ============================================================================
# Terraform Configuration for AI Service (GCP GKE + GPU)
# Project: ai-service-air-icon (중고차 판매 특화 AI 서비스)
# CSP: Google Cloud Platform (GCP)
# ============================================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }

  backend "gcs" {
    bucket = "ai-service-terraform-state"
    prefix = "terraform/state"
  }
}

# ============================================================================
# Variables
# ============================================================================

variable "project_id" {
  description = "GCP 프로젝트 ID"
  type        = string
  default     = "ai-service-air-icon"
}

variable "region" {
  description = "GCP 리전 (서울)"
  type        = string
  default     = "asia-northeast3"
}

variable "zone" {
  description = "GCP 존"
  type        = string
  default     = "asia-northeast3-a"
}

variable "cluster_name" {
  description = "GKE 클러스터 이름"
  type        = string
  default     = "ai-service-cluster"
}

variable "gpu_node_count_min" {
  description = "GPU 노드 최소 개수"
  type        = number
  default     = 1
}

variable "gpu_node_count_max" {
  description = "GPU 노드 최대 개수"
  type        = number
  default     = 5
}

variable "gpu_machine_type" {
  description = "GPU 머신 타입"
  type        = string
  default     = "g2-standard-8"
}

variable "gpu_type" {
  description = "GPU 타입"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "노드당 GPU 개수"
  type        = number
  default     = 1
}

variable "disk_size_gb" {
  description = "노드 디스크 크기 (GB)"
  type        = number
  default     = 200
}

variable "disk_type" {
  description = "디스크 타입"
  type        = string
  default     = "pd-ssd"
}

variable "huggingface_token" {
  description = "Hugging Face API 토큰"
  type        = string
  sensitive   = true
}

variable "environment" {
  description = "환경 (production/staging/development)"
  type        = string
  default     = "production"
}

variable "storage_bucket_name" {
  description = "모델 저장용 GCS 버킷 이름"
  type        = string
  default     = "ai-service-models-backup"
}

# ============================================================================
# Provider Configuration
# ============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

# ============================================================================
# VPC Network
# ============================================================================

resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  description = "VPC for AI service infrastructure"
}

resource "google_compute_subnetwork" "private_subnet" {
  name          = "${var.cluster_name}-private-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }

  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_subnetwork" "public_subnet" {
  name          = "${var.cluster_name}-public-subnet"
  ip_cidr_range = "10.0.2.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

# ============================================================================
# Cloud NAT (Outbound Internet Access)
# ============================================================================

resource "google_compute_router" "router" {
  name    = "${var.cluster_name}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name   = "${var.cluster_name}-nat"
  router = google_compute_router.router.name
  region = var.region

  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ============================================================================
# Firewall Rules
# ============================================================================

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_firewall" "allow_health_check" {
  name    = "${var.cluster_name}-allow-health-check"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "8080", "443"]
  }

  source_ranges = ["35.191.0.0/16", "130.211.0.0/22"]
  target_tags   = ["gke-node"]
}

# ============================================================================
# GKE Cluster
# ============================================================================

resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.private_subnet.name

  # IP allocation for pods and services
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity (IAM integration)
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Network Policy
  network_policy {
    enabled  = true
    provider = "PROVIDER_UNSPECIFIED"
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Monitoring and logging
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
    managed_prometheus {
      enabled = true
    }
  }

  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All networks"
    }
  }

  lifecycle {
    ignore_changes = [node_pool, initial_node_count]
  }
}

# ============================================================================
# GPU Node Pool (L4 24GB)
# ============================================================================

resource "google_container_node_pool" "gpu_nodes" {
  name       = "${var.cluster_name}-gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.gpu_node_count_min

  autoscaling {
    min_node_count = var.gpu_node_count_min
    max_node_count = var.gpu_node_count_max
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = var.gpu_machine_type
    disk_size_gb = var.disk_size_gb
    disk_type    = var.disk_type
    image_type   = "COS_CONTAINERD"

    # GPU configuration
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    # Spot instances for cost optimization
    spot = true

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]

    labels = {
      workload-type = "gpu"
      environment   = var.environment
      project       = "ai-service-air-icon"
      managed_by    = "terraform"
    }

    tags = ["gke-node", "${var.cluster_name}-node"]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  lifecycle {
    ignore_changes = [node_count]
  }
}

# ============================================================================
# CPU Node Pool (Optional - for API Gateway)
# ============================================================================

resource "google_container_node_pool" "cpu_nodes" {
  name       = "${var.cluster_name}-cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = 2

  autoscaling {
    min_node_count = 2
    max_node_count = 4
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-standard"
    image_type   = "COS_CONTAINERD"

    spot = true

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      workload-type = "cpu"
      environment   = var.environment
      managed_by    = "terraform"
    }

    tags = ["gke-node", "${var.cluster_name}-node"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

# ============================================================================
# Persistent Disk for Model Storage
# ============================================================================

resource "google_compute_disk" "model_storage" {
  name  = "${var.cluster_name}-model-storage"
  type  = "pd-ssd"
  zone  = var.zone
  size  = 200

  labels = {
    environment = var.environment
    project     = "ai-infra"
    managed_by  = "terraform"
  }

  physical_block_size_bytes = 4096
}

# ============================================================================
# Cloud Storage Bucket (Model Backup)
# ============================================================================

resource "google_storage_bucket" "model_backup" {
  name          = "${var.project_id}-${var.storage_bucket_name}"
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  labels = {
    environment = var.environment
    project     = "ai-infra"
    managed_by  = "terraform"
  }
}

# ============================================================================
# Secret Manager (Hugging Face Token)
# ============================================================================

resource "google_secret_manager_secret" "huggingface_token" {
  secret_id = "huggingface-token"

  replication {
    auto {}
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "google_secret_manager_secret_version" "huggingface_token_version" {
  secret      = google_secret_manager_secret.huggingface_token.id
  secret_data = var.huggingface_token
}

# ============================================================================
# Static IP for Load Balancer
# ============================================================================

resource "google_compute_global_address" "lb_ip" {
  name         = "${var.cluster_name}-lb-ip"
  address_type = "EXTERNAL"
}

# ============================================================================
# Cloud Armor Security Policy
# ============================================================================

resource "google_compute_security_policy" "policy" {
  name        = "${var.cluster_name}-security-policy"
  description = "DDoS protection and WAF rules"

  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }

  # Rate limiting
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
  }

  # Block SQL injection
  rule {
    action   = "deny(403)"
    priority = "2000"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('sqli-stable')"
      }
    }
  }

  # Block XSS
  rule {
    action   = "deny(403)"
    priority = "3000"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-stable')"
      }
    }
  }
}

# ============================================================================
# Artifact Registry (Container Images)
# ============================================================================

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "ai-service"
  description   = "Docker repository for AI service containers"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ============================================================================
# Service Account for Workload Identity
# ============================================================================

resource "google_service_account" "ai_service_sa" {
  account_id   = "${var.cluster_name}-sa"
  display_name = "Service Account for AI Service"
  description  = "Used by AI service pods to access GCP resources"
}

resource "google_project_iam_member" "ai_service_storage_access" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.ai_service_sa.email}"
}

resource "google_project_iam_member" "ai_service_secret_access" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.ai_service_sa.email}"
}

resource "google_service_account_iam_binding" "workload_identity_binding" {
  service_account_id = google_service_account.ai_service_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/ai-service-sa]",
  ]
}

# ============================================================================
# Kubernetes Provider Configuration
# ============================================================================

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth[0].cluster_ca_certificate)
}

# ============================================================================
# Kubernetes Namespace
# ============================================================================

resource "kubernetes_namespace" "ai_service" {
  metadata {
    name = "ai-service"
    labels = {
      name        = "ai-service"
      environment = var.environment
      managed_by  = "terraform"
    }
  }

  depends_on = [google_container_cluster.primary]
}

# ============================================================================
# Kubernetes Service Account
# ============================================================================

resource "kubernetes_service_account" "ai_service" {
  metadata {
    name      = "ai-service-sa"
    namespace = kubernetes_namespace.ai_service.metadata[0].name
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.ai_service_sa.email
    }
  }
}

# ============================================================================
# Outputs
# ============================================================================

output "cluster_name" {
  description = "GKE 클러스터 이름"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE 클러스터 엔드포인트"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "클러스터 CA 인증서"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "vpc_name" {
  description = "VPC 네트워크 이름"
  value       = google_compute_network.vpc.name
}

output "private_subnet_name" {
  description = "프라이빗 서브넷 이름"
  value       = google_compute_subnetwork.private_subnet.name
}

output "load_balancer_ip" {
  description = "로드 밸런서 외부 IP"
  value       = google_compute_global_address.lb_ip.address
}

output "storage_bucket_name" {
  description = "모델 백업 GCS 버킷 이름"
  value       = google_storage_bucket.model_backup.name
}

output "artifact_registry_url" {
  description = "Artifact Registry URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "service_account_email" {
  description = "AI 서비스 서비스 계정 이메일"
  value       = google_service_account.ai_service_sa.email
}

output "gpu_node_pool_name" {
  description = "GPU 노드 풀 이름"
  value       = google_container_node_pool.gpu_nodes.name
}

output "secret_manager_secret_id" {
  description = "Secret Manager 시크릿 ID"
  value       = google_secret_manager_secret.huggingface_token.secret_id
}

output "connect_command" {
  description = "클러스터 연결 명령어"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${var.region} --project ${var.project_id}"
}

# ============================================================================
# Local Variables for Tags
# ============================================================================

locals {
  common_labels = {
    project     = "ai-infra"
    environment = var.environment
    managed_by  = "terraform"
    service     = "ai-service-air-icon"
  }
}
