// Jenkinsfile (Declarative Pipeline)

pipeline {
    agent any
    environment {
        // DOCKER_REGISTRY_USER = 'tructran172003' 
        // Tên các ảnh Docker
        TTS_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-tts-service"
        STT_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-stt-service"
        CHATBOT_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/rag-chatbot-service"
        // DB_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/db"

        // ID của registry credential đã được lưu trong Jenkins
        DOCKER_CREDENTIALS_ID = 'dockerhub-credentials'
        ENV_CREDENTIALS_ID = 'rag-project-env-file'

        // SỬ DỤNG BUILD_NUMBER LÀM TAG PHIÊN BẢN
        IMAGE_TAG = "${env.BUILD_NUMBER}"
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
            when {
                changeset "/src/services/**"
            }
            steps {
                script {
                    // Sử dụng withCredentials để tải secret vào một biến môi trường tạm thời
                    withCredentials([string(credentialsId: ENV_CREDENTIALS_ID, variable: 'ENV_FILE_CONTENT')]) {
                        echo "Tạo file .env từ Jenkins Credentials..."
                        
                        // Tạo thư mục 'config' nếu nó chưa tồn tại
                        sh 'mkdir -p config'
                        
                        // Ghi toàn bộ nội dung đã lưu vào file config/.env
                        sh 'echo "${ENV_FILE_CONTENT}" > config/.env'

                        // --- BƯỚC DEBUGGING MỚI ---
                        echo "--- Bắt đầu kiểm tra file .env ---"
                        echo "Nội dung của file config/.env được tạo ra:"
                        sh 'cat config/.env'
                        echo "Kiểm tra sự tồn tại của biến SIMILARITY_THRESHOLD_FOR_MERGE:"
                        // Lệnh grep sẽ trả về exit code 0 nếu tìm thấy, và 1 nếu không.
                        // || true để đảm bảo pipeline không bị lỗi nếu không tìm thấy.
                        sh 'grep SIMILARITY_THRESHOLD_FOR_MERGE config/.env || echo "CẢNH BÁO: Biến SIMILARITY_THRESHOLD_FOR_MERGE không tìm thấy trong file .env!"'
                        echo "--- Kết thúc kiểm tra file .env ---"

                        // --- BƯỚC DEBUGGING MỚI ---
                        echo "Kiểm tra giá trị của biến môi trường từ bên trong shell của container..."
                        sh 'echo "Giá trị của SIMILARITY_THRESHOLD_FOR_MERGE là: $SIMILARITY_THRESHOLD_FOR_MERGE"'

                        echo "Bắt đầu xây dựng các ảnh Docker..."
                        sh 'docker compose -f infrastructure/docker/docker-compose.yml build'

                        echo "Khởi động các dịch vụ ở chế độ nền..."
                        sh 'docker compose -f infrastructure/docker/docker-compose.yml up -d'
                        
                        echo "Cài đặt các thư viện và chạy embedding bên trong container..."
                        // SỬA LỖI: Sử dụng --env-file để nạp trực tiếp các biến môi trường
                        docker.image('nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04').inside("--user root --env-file ${pwd()}/config/.env") {
                            sh '''
                                set -e
                                # đảm bảo thư mục tồn tại và quyền sở hữu hợp lý
                                mkdir -p /var/lib/apt/lists/partial
                                chown -R root:root /var/lib/apt/lists || true

                                apt-get update
                                apt-get install -y --no-install-recommends build-essential python3-dev git python3-pip 
                                rm -rf /var/lib/apt/lists/*

                                python3 -m pip install --upgrade pip
                                python3 -m pip install -r requirements.txt

                                export $(grep -v '^#' config/.env | xargs)
                                python3 src/core/embedding.py
                            '''
                        }
                        echo "Hoàn tất giai đoạn Build, Run và Ingest."
                    }
                }
            }
        }


        // Giai đoạn 3: Chạy các bài test bên trong một Docker container
        stage('Run Tests') {
            steps {
                script {
                    echo "Chuẩn bị môi trường test sử dụng Docker..."
                    docker.image('nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04').inside('--user root --network elk-net') {
                        echo "Bắt đầu chạy các bài kiểm thử (pytest) bên trong container..."
                        
                        // Cài đặt các phụ thuộc cần thiết cho test nếu có
                        sh 'apt-get update && apt-get install -y --no-install-recommends \
                            build-essential \
                            python3-dev \
                            git \
                            python3-pip \
                            espeak \
                            libpq-dev \
                            gcc \
                            && rm -rf /var/lib/apt/lists/*'
                        sh 'python3 -m pip install --upgrade pip'
                        sh 'python3 -m pip install -r requirements.txt'
                        // sh 'mkdir -p ./StyleTTS2/Utils/ASR'
                        // sh 'pip install --upgrade gdown -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=120 --retries=5'
                        // sh 'gdown 1Yx92zfeAjdsh5wddji8vrqpZdGw1eyrN \
                        //     -O ./StyleTTS2/Utils/ASR/epoch_00080.pth'

                        // Chạy các lệnh test trực tiếp
                        sh '''
                            set -e
                            echo "Running tests..."
                            python3 -m pytest tests/cicd/test_tts_api.py
                            python3 -m pytest tests/cicd/test_whisper-api.py
                            python3 -m pytest tests/cicd/test_chatbot_api.py
                        '''
                        
                        echo "Tất cả các bài kiểm thử đã pass."
                    }
                }
            }
        }

        // Giai đoạn 4: Đẩy ảnh Docker lên Registry
        stage('Push Docker Images') {
            when {
                changeset 'Container_Folder/**'
            }
            steps {
                script {
                    echo "Đang đẩy các ảnh Docker lên Docker Hub..."
                    
                    // Đăng nhập vào Docker Hub sử dụng credential đã lưu trong Jenkins
                    docker.withRegistry("https://registry.hub.docker.com", DOCKER_CREDENTIALS_ID) {
                        
                        // Đẩy từng ảnh với tag phiên bản cụ thể
                        docker.image("${TTS_IMAGE_NAME}:${IMAGE_TAG}").push()
                        docker.image("${STT_IMAGE_NAME}:${IMAGE_TAG}").push()
                        docker.image("${CHATBOT_IMAGE_NAME}:${IMAGE_TAG}").push()

                        // Đẩy thêm tag 'latest' để trỏ đến phiên bản mới nhất
                        docker.image("${TTS_IMAGE_NAME}:${IMAGE_TAG}").push("latest")
                        docker.image("${STT_IMAGE_NAME}:${IMAGE_TAG}").push("latest")
                        docker.image("${CHATBOT_IMAGE_NAME}:${IMAGE_TAG}").push("latest")

                        sh 'docker compose -f infrastructure/docker/docker-compose.jenkins.yml push'
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
            echo 'Dọn dẹp workspace...'
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
