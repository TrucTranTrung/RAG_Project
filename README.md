# Overview
- This RAG project focuses on psychology and aims to generate responses that closely resemble real-world human interaction.
- The system includes both Text-to-Speech (TTS) and Speech-to-Text (STT) modules, enabling fully voice-based communication.
- The entire application is containerized with Docker and deployed through a CI/CD pipeline using Jenkins.
- For monitoring, logging, and observability, the infrastructure integrates Grafana, Prometheus, and the ELK stack.

## Front-end
<img width="1851" height="922" alt="Screenshot from 2025-11-21 14-46-32" src="https://github.com/user-attachments/assets/b6e6cb66-66c7-4813-a044-83598d33a282" />

## Monitoring
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/acfaeeb5-7a49-44b2-a5d2-e7656ff57347" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/477ce692-c64d-4f12-b216-7183a0881911" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/a8483253-9f01-4875-a267-c92c644d7e7b" width="300"></td>
  </tr>
</table>


weight : https://drive.google.com/drive/folders/1njve-dILpn-wqR32L7Yqk1wwMN2ADfl8?usp=sharing

# Run Docker Compose
docker network create elk-net
docker compose -f infrastructure/docker/docker-compose-monitor.yml up
docker compose -f infrastructure/docker/docker-compose.yml up 

# Run Docker ELK Compose
docker compose -f infrastructure/docker/docker-compose.elk.yml up 

# Chạy minikube
minikube start --driver=docker --gpus=all



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

# Chạy Streamlit
streamlit run src/services/Front_end/chatbot_app.py
