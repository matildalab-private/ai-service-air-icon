provider "google" {
  project = "ai-service-project"
  region  = "asia-northeast3"  # 서울 리전
}

# VPC 네트워크 생성
resource "google_compute_network" "vpc_network" {
  name                    = "ai-service-vpc"
  auto_create_subnetworks = false
}

# 서브넷 생성
resource "google_compute_subnetwork" "subnet" {
  name          = "ai-service-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = "asia-northeast3"
  network       = google_compute_network.vpc_network.id
}

# 방화벽 규칙 생성
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = google_compute_network.vpc_network.id
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ai-server"]
}

resource "google_compute_firewall" "allow_web" {
  name    = "allow-web"
  network = google_compute_network.vpc_network.id
  
  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ai-server"]
}

# 고정 IP 주소 생성
resource "google_compute_address" "static_ip" {
  name   = "ai-service-static-ip"
  region = "asia-northeast3"
}

# 부팅 디스크를 위한 이미지
data "google_compute_image" "ubuntu" {
  family  = "ubuntu-2004-lts"
  project = "ubuntu-os-cloud"
}

# GPU 인스턴스 생성
resource "google_compute_instance" "ai_service_instance" {
  name         = "ai-service-instance"
  machine_type = "g2-standard-8"  # 32GB RAM과 GPU 지원
  zone         = "asia-northeast3-a"
  tags         = ["ai-server"]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.ubuntu.self_link
      size  = 100  # 100GB
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.subnet.self_link
    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  # GPU 설정
  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"  # GPU 인스턴스는 라이브 마이그레이션을 지원하지 않음
    automatic_restart   = true
    preemptible         = false
  }

  # GPU 드라이버 및 환경 설정을 위한 시작 스크립트
  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y git python3-pip
    
    # Install CUDA drivers
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    apt-get update
    apt-get install -y cuda
    
    # Clone repo and setup
    git clone https://github.com/matildalab-private/ai-service-air-icon.git /opt/ai-service
    cd /opt/ai-service
    pip3 install torch transformers huggingface_hub pandas datasets python-dotenv
    
    # Create service file
    cat > /etc/systemd/system/aiservice.service << 'EOL'
    [Unit]
    Description=AI Service for Used Car Sales
    After=network.target
    
    [Service]
    User=root
    WorkingDirectory=/opt/ai-service
    ExecStart=/usr/bin/python3 app.py
    Restart=always
    
    [Install]
    WantedBy=multi-user.target
    EOL
    
    systemctl daemon-reload
    systemctl enable aiservice
    systemctl start aiservice
  EOF

  # 태그 설정
  labels = {
    project     = "ai-infra"
    environment = "production"
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

# 클라우드 스토리지 버킷 - 모델 및 데이터 저장용
resource "google_storage_bucket" "model_storage" {
  name          = "ai-service-model-storage"
  location      = "ASIA-NORTHEAST3"
  storage_class = "STANDARD"
  force_destroy = true

  labels = {
    project     = "ai-infra"
    environment = "production"
  }

  versioning {
    enabled = true
  }
}

# 모니터링 알림 정책 - CPU 사용량
resource "google_monitoring_alert_policy" "cpu_usage_alert" {
  display_name = "High CPU Usage Alert"
  combiner     = "OR"
  
  conditions {
    display_name = "CPU usage above 80%"
    
    condition_threshold {
      filter          = "resource.type = \"gce_instance\" AND resource.labels.instance_id = \"${google_compute_instance.ai_service_instance.instance_id}\" AND metric.type = \"compute.googleapis.com/instance/cpu/utilization\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }
  
  notification_channels = []
}

# 출력값
output "instance_ip" {
  value = google_compute_address.static_ip.address
}

output "model_storage_url" {
  value = google_storage_bucket.model_storage.url
}