# Overview
- This RAG project focuses on psychology and aims to generate responses that closely resemble real-world human interaction.
- The system includes both Text-to-Speech (TTS) and Speech-to-Text (STT) modules, enabling fully voice-based communication.
- The entire application is containerized with Docker and deployed through a CI/CD pipeline using Jenkins.
- For monitoring, logging, and observability, the infrastructure integrates Grafana, Prometheus, and the ELK stack.
- Alert: This project required GPU for model

## Front-end
<img width="1851" height="922" alt="Screenshot from 2025-11-21 14-46-32" src="https://github.com/user-attachments/assets/b6e6cb66-66c7-4813-a044-83598d33a282" />

## Monitoring Prometheus
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/acfaeeb5-7a49-44b2-a5d2-e7656ff57347" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/477ce692-c64d-4f12-b216-7183a0881911" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/a8483253-9f01-4875-a267-c92c644d7e7b" width="300"></td>
  </tr>
</table>

## Monitoring Logs
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/acfaeeb5-7a49-44b2-a5d2-e7656ff57347" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/a8483253-9f01-4875-a267-c92c644d7e7b" width="300"></td>
  </tr>
</table>

## CI/CD Jenskin
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/acfaeeb5-7a49-44b2-a5d2-e7656ff57347" width="300"></td>
  </tr>
</table>



# Prepare
```bash
git clone https://github.com/TrucTranTrung/RAG_Project
cd RAG_Project

conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

- Access to this drive: https://drive.google.com/drive/folders/1byAlMsILagpUjIbB3LWUPYyTCfkzdYx3?usp=sharing
- download weight: epoch_00080.pth and put it in src/services/Text_to_Speech/StyleTTS2/Utils/ASR/ folder.
- download weight: epoch_2nd_00100.pth and put it in /StyleTTS2/Models/LJSpeech/ folder.

# Run Docker Compose
```bash
docker network create elk-net
docker compose -f infrastructure/docker/docker-compose.yml up
```

- Frontend is at: http://localhost:9001/static/index.html

## Front-end
<img width="1851" height="922" alt="Screenshot from 2025-11-21 14-46-32" src="https://github.com/user-attachments/assets/b6e6cb66-66c7-4813-a044-83598d33a282" />

# Run Docker ELK Compose for logs monitor
docker compose -f infrastructure/docker/docker-compose.elk.yml up

# Run Docker Monitor Compose for monitor GPU, CPU....
docker compose -f infrastructure/docker/docker-compose-monitor.yml up

# Run Docker Jenskin Compose for CI/CD
docker compose -f infrastructure/docker/docker-compose.jenkins.yml up

# Chạy minikube
minikube start --driver=docker --gpus=all

http://localhost:9001/static/index.html
# Tạo namespace mới
kubectl create namespace rag-app

# Tạo secret
kubectl create secret generic rag-app-secrets --from-env-file=config/.env -n rag-app

# Kiểm tra secret
kubectl get secrets -n rag-app

# Khởi tạo
kubectl apply -f deployment.yml
kubectl get pods -n rag-app
kubectl describe pod ai-server-deployment-784fcfccd4-fc5q6 -n rag-app