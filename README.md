# Overview
- This RAG project focuses on psychology and aims to generate responses that closely resemble real-world human interaction.
- The system includes both Text-to-Speech (TTS) and Speech-to-Text (STT) modules, enabling fully voice-based communication.
- The entire application is containerized with Docker and deployed through a CI/CD pipeline using Jenkins.
- For monitoring, logging, and observability, the infrastructure integrates Grafana, Prometheus, and the ELK stack.
- Alert: This project required GPU for model

## Project Pipeline
<img width="1219" height="712" alt="Screenshot from 2026-04-30 21-20-18" src="https://github.com/user-attachments/assets/07387c38-2f68-4bf6-9e3b-90e4df13e19e" />


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

# Push Data to PG Vector
```bash
conda activate rag_env
python src/core/embedding.py
```
For small or busy GPUs, lower the embedding workload before running:
```bash
export EMBEDDING_BATCH_SIZE=4
export PGVECTOR_INSERT_BATCH_SIZE=4
python src/core/embedding.py
```
Use `EMBEDDING_DEVICE=cpu` if CUDA memory is still not available.

- Frontend is at: http://localhost:9001/static/index.html

# Chat Model Provider
The chatbot can run without OpenAI/Gemini quota for demos:
```bash
export CHAT_MODEL_PROVIDER=local
```
This returns a simple answer from retrieved PGVector context and does not load a local LLM.
To use OpenAI again:
```bash
export CHAT_MODEL_PROVIDER=openai
export OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Front-end
<img width="1851" height="922" alt="Screenshot from 2025-11-21 14-46-32" src="https://github.com/user-attachments/assets/b6e6cb66-66c7-4813-a044-83598d33a282" />


# Run Docker ELK Compose for logs monitor
```bash
docker compose -f infrastructure/docker/docker-compose.elk.yml up
```

- Kibana port is 5601, you can access to query at: localhost:5601
- Access Observability -> Stream to query logs in container
  
## Monitoring Logs
<img width="1846" height="881" alt="Screenshot from 2025-12-10 16-15-02" src="https://github.com/user-attachments/assets/0cf12738-029b-4d31-a641-845b752e82f1" />

# Run Docker Monitor Compose for monitor GPU, CPU....
```bash
docker compose -f infrastructure/docker/docker-compose-monitor.yml up
```

- Granfa port is 3000, you can access to query at: localhost:3000
- Grafana has 3 dashboards: DCGM for GPU, node-exporter for CPU and cadvisor for container

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/acfaeeb5-7a49-44b2-a5d2-e7656ff57347" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/477ce692-c64d-4f12-b216-7183a0881911" width="300"></td>
    <td><img src="https://github.com/user-attachments/assets/a8483253-9f01-4875-a267-c92c644d7e7b" width="300"></td>
  </tr>
</table>

- Access Jeager at localhost:16686
- Choose service for tracing

## Jeager Tracing
<img width="1844" height="930" alt="Screenshot from 2025-12-10 16-23-58" src="https://github.com/user-attachments/assets/b45d63c6-a510-4529-95a6-c2cdeedc52ff" />

# Run Docker Jenskin Compose for CI/CD
```bash
docker compose -f infrastructure/docker/docker-compose.jenkins.yml up
```

- Jenkins port is 8080, you can access to query at: localhost:8080

## CI/CD Jenskin
<img width="1851" height="922" alt="Screenshot from 2025-11-21 14-46-32" src="https://github.com/user-attachments/assets/5f1d08ce-bc2c-451c-807f-d7b7cfd449d0" />

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
