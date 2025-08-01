# RAG_Project

weight : https://drive.google.com/drive/folders/1njve-dILpn-wqR32L7Yqk1wwMN2ADfl8?usp=sharing

# Chạy minikube
minikube start --gpus=all

kubectl apply -f time-slicing-config.yml
kubectl apply -f nvidia-plugin-with-config.yml

# Tạo namespace mới
kubectl create namespace rag-app

# Tạo secret
kubectl create secret generic rag-app-secrets --from-env-file=config/.env -n rag-app

# Kiểm tra secret
kubectl get secrets -n rag-app

# Khởi tạo
kubectl apply -f deployment.yml