# RAG_Project

weight : https://drive.google.com/drive/folders/1njve-dILpn-wqR32L7Yqk1wwMN2ADfl8?usp=sharing


# Tạo namespace mới
kubectl create namespace rag-app

# Tạo secret
kubectl create secret generic my-secret --from-literal=username=admin --namespace=rag-app

# Kiểm tra secret
kubectl get secrets -n rag-app

# Khởi tạo
kubectl apply -f deployment.yml