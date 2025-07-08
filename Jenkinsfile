// Jenkinsfile (Declarative Pipeline)

pipeline {
    agent any
    environment {
        // DOCKER_REGISTRY_USER = 'tructran172003' 
        
        // Tên các ảnh Docker
        // TTS_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-tts-service"
        // STT_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-stt-service"
        // CHATBOT_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-chatbot-service"
        // DB_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/db"

        // ID của registry credential đã được lưu trong Jenkins
        DOCKER_CREDENTIALS_ID = 'dockerhub-credentials'
        ENV_CREDENTIALS_ID = 'rag-project-env-file'

        // SỬ DỤNG BUILD_NUMBER LÀM TAG PHIÊN BẢN
        // IMAGE_TAG = "${env.BUILD_NUMBER}"
    }

    stages {
        // Giai đoạn 1: Lấy mã nguồn từ Git
        stage('Checkout SCM') {
            steps {
                echo 'Đang lấy mã nguồn từ Git...'
                git url: 'https://github.com/TrucTranTrung/RAG_Project', branch: 'main'
            }
        }

        // Giai đoạn 2: Xây dựng các ảnh Docker
        stage('Build Docker Images') {
            // VÍ DỤ: Thêm 'when' vào đây nếu bạn muốn stage này chỉ chạy khi có thay đổi
            when {
                changeset "Container_Folder/**"
            }
            steps {
                script {
                    // Sử dụng withCredentials để tải secret vào một biến môi trường tạm thời
                    withCredentials([string(credentialsId: ENV_CREDENTIALS_ID, variable: 'ENV_FILE_CONTENT')]) {
                        echo "Tạo file .env từ Jenkins Credentials..."
                        
                        // SỬA LỖI: Tạo thư mục 'config' nếu nó chưa tồn tại
                        sh 'mkdir -p config'
                        
                        // Ghi toàn bộ nội dung đã lưu vào file config/.env
                        // Dấu ngoặc kép quanh ${ENV_FILE_CONTENT} rất quan trọng để giữ nguyên định dạng nhiều dòng
                        sh 'echo "${ENV_FILE_CONTENT}" > config/.env'

                        echo "Bắt đầu xây dựng các ảnh Docker và khởi động dịch vụ..."
                        sh 'docker compose -f docker-compose.jenkins.yml build'
                        sh 'docker compose -f docker-compose.jenkins.yml up -d'
                        
                        echo "Cài đặt các thư viện cần thiết và chạy embedding..."
                        docker.image('nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04').inside {
                            sh 'apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev git python3-pip && rm -rf /var/lib/apt/lists/*'
                            sh 'pip install -r requirements.txt'
                            sh 'python3 embedding.py'
                        }

                        echo "Hoàn tất giai đoạn Build và Run."
                    }
                }
            }
        }

        // Giai đoạn 3: Chạy các bài test bên trong một Docker container
        stage('Run Tests') {
            steps {
                script {
                    echo "Chuẩn bị môi trường test sử dụng Docker..."
                    docker.image('python:3.10-slim').inside {
                        echo "Bắt đầu chạy các bài kiểm thử (pytest) bên trong container..."
                        sh 'chmod +x ./run_test.sh'
                        // Chạy các file test
                        sh './run_test.sh'
                        
                        echo "Tất cả các bài kiểm thử đã pass."
                    }
                }
            }
        }

        // Giai đoạn 4: Đẩy ảnh Docker lên Registry
        stage('Push Docker Images') {
            when {
                branch 'main'
                // Bạn cũng có thể thêm điều kiện changeset ở đây
                // changeset "Container_Folder/**"
            }
            steps {
                script {
                    echo "Đang đẩy các ảnh Docker lên Docker Hub..."
                    
                    // Đăng nhập vào Docker Hub sử dụng credential đã lưu trong Jenkins
                    docker.withRegistry("https://registry.hub.docker.com", DOCKER_CREDENTIALS_ID) {
                        
                        // Đẩy từng ảnh với tag phiên bản cụ thể
                        // docker.image("${TTS_IMAGE_NAME}:${IMAGE_TAG}").push()
                        // docker.image("${STT_IMAGE_NAME}:${IMAGE_TAG}").push()
                        // docker.image("${CHATBOT_IMAGE_NAME}:${IMAGE_TAG}").push()

                        // Đẩy thêm tag 'latest' để trỏ đến phiên bản mới nhất
                        // docker.image("${TTS_IMAGE_NAME}:${IMAGE_TAG}").push("latest")
                        // docker.image("${STT_IMAGE_NAME}:${IMAGE_TAG}").push("latest")
                        // docker.image("${CHATBOT_IMAGE_NAME}:${IMAGE_TAG}").push("latest")

                        sh 'docker compose -f docker-compose.jenkins.yml push'
                    }
                    
                    echo "Đẩy ảnh Docker hoàn tất."
                }
            }
        }
        
        // Giai đoạn 5: Triển khai ứng dụng
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo "Bắt đầu triển khai ứng dụng phiên bản: ${IMAGE_TAG}"
                
                // placeholder.

                echo "Triển khai hoàn tất (placeholder)."
            }
        }
    }
    
    // Các hành động sẽ được thực hiện sau khi pipeline kết thúc
    post {
        always {
            echo 'Pipeline đã kết thúc.'
            cleanWs()
        }
        success {
            echo 'Pipeline thành công!'
        }
        failure {
            echo 'Pipeline thất bại!'
        }
    }
}
