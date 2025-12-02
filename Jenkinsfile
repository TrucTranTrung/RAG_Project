pipeline {
    agent any
    environment {
        DOCKER_REGISTRY_USER = 'tructran172003'

        // Tên các ảnh Docker khớp với docker-compose của bạn
        TTS_IMAGE_NAME     = "${env.DOCKER_REGISTRY_USER}/tts-api"
        STT_IMAGE_NAME     = "${env.DOCKER_REGISTRY_USER}/whisper-api"
        CHATBOT_IMAGE_NAME = "${env.DOCKER_REGISTRY_USER}/chatbot-api"

        // ID của registry credential đã được lưu trong Jenkins
        DOCKER_CREDENTIALS_ID = 'dockerhub-credentials'
        ENV_CREDENTIALS_ID    = 'rag-project-env-file'

        // SỬ DỤNG BUILD_NUMBER LÀM TAG PHIÊN BẢN
        IMAGE_TAG = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Checkout SCM') {
            steps {
                echo 'Đang lấy mã nguồn từ Git...'
                git url: 'https://github.com/TrucTranTrung/RAG_Project', branch: 'main'
            }
        }

        stage('Ensure model (download from Google Drive)') {
            steps {
                script {
                    def FILE_ID = '1Yx92zfeAjdsh5wddji8vrqpZdGw1eyrN'
                    def OUT = 'src/services/Text_to_Speech/StyleTTS2/Utils/ASR/epoch_00080.pth'

                    sh """
                      set -e
                      echo 'Ensure model path and parent exist...'
                      mkdir -p \$(dirname ${OUT})

                      if [ -f ${OUT} ]; then
                        echo "Model already exists at ${OUT} (size: \$(stat -c%s ${OUT}) bytes). Skipping download."
                        exit 0
                      fi

                      echo "Downloading model from Google Drive (id=${FILE_ID}) into ${OUT} ..."
                      # fetch initial page to get potential confirm token (for large files)
                      curl -c /tmp/gdcookies -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o /tmp/gdpage.html || true

                      # try to extract confirm token
                      CONFIRM=\$(grep -o 'confirm=[0-9A-Za-z_-]*' /tmp/gdpage.html | sed 's/confirm=//' | head -n1 || true)
                      if [ -z "\$CONFIRM" ]; then
                        CONFIRM=\$(sed -n 's/.*confirm=\\([^;&]*\\).*/\\1/p' /tmp/gdpage.html | head -n1 || true)
                      fi

                      if [ -n "\$CONFIRM" ]; then
                        echo "Found confirm token: \$CONFIRM — performing confirmed download..."
                        curl -Lb /tmp/gdcookies "https://drive.google.com/uc?export=download&confirm=\${CONFIRM}&id=${FILE_ID}" -o ${OUT}
                      else
                        echo "No confirm token found — trying direct download (works for small files)..."
                        curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o ${OUT}
                      fi

                      # verify
                      if [ ! -f ${OUT} ] || [ \$(stat -c%s ${OUT}) -lt 100 ]; then
                        echo "ERROR: Download failed or file too small. Show page for debug:"
                        sed -n '1,200p' /tmp/gdpage.html || true
                        ls -la \$(dirname ${OUT}) || true
                        exit 1
                      fi

                      # set permisssions
                      chmod 644 ${OUT}
                      echo "Downloaded model OK: \$(stat -c'%n %s bytes' ${OUT})"
                    """
                }
            }
        }

        stage('Prepare .env and Start Services') {
            when {
                changeset "src/services/**"
            }
            steps {
                script {
                    withCredentials([string(credentialsId: ENV_CREDENTIALS_ID, variable: 'ENV_FILE_CONTENT')]) {
                        sh 'mkdir -p config'
                        sh 'echo "${ENV_FILE_CONTENT}" > config/.env'
                        echo "Nội dung config/.env (first 50 lines):"
                        sh 'sed -n "1,50p" config/.env || true'

                        // build local (compose sẽ dùng file .env để inject env variables)
                        sh 'docker compose --env-file config/.env -f infrastructure/docker/docker-compose.yml up -d --build'
                    }
                }
            }
        }

        stage('Build Docker Images (tag from compose)') {
            when {
                changeset "src/services/**"
            }
            steps {
                script {
                    echo "Đã build bằng docker compose ở stage trước — sẽ tag lại các image với ${IMAGE_TAG}"

                    // Tên image theo docker-compose 
                    def ttsOrig = "${TTS_IMAGE_NAME}:v1.0"       
                    def sttOrig = "${STT_IMAGE_NAME}:v1.0"
                    def chatbotOrig = "${CHATBOT_IMAGE_NAME}:v1.0"

                    // Tag lại với IMAGE_TAG và latest
                    sh "docker tag ${ttsOrig} ${TTS_IMAGE_NAME}:${IMAGE_TAG} || true"
                    sh "docker tag ${ttsOrig} ${TTS_IMAGE_NAME}:latest || true"

                    sh "docker tag ${sttOrig} ${STT_IMAGE_NAME}:${IMAGE_TAG} || true"
                    sh "docker tag ${sttOrig} ${STT_IMAGE_NAME}:latest || true"

                    sh "docker tag ${chatbotOrig} ${CHATBOT_IMAGE_NAME}:${IMAGE_TAG} || true"
                    sh "docker tag ${chatbotOrig} ${CHATBOT_IMAGE_NAME}:latest || true"

                    echo "Tagging hoàn tất."
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    echo "Chuẩn bị môi trường test sử dụng Docker..."
                    docker.image('nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04').inside('--user root --network elk-net') {
                        sh 'apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev git python3-pip espeak libpq-dev gcc && rm -rf /var/lib/apt/lists/*'
                        sh 'python3 -m pip install --upgrade pip'
                        sh 'python3 -m pip install -r requirements.txt || true'

                        sh '''
                            set -e
                            echo "Running tests..."
                            python3 -m pytest tests/cicd/test_tts_api.py || true
                            python3 -m pytest tests/cicd/test_whisper-api.py || true
                            python3 -m pytest tests/cicd/test_chatbot_api.py || true
                        '''
                    }
                }
            }
        }

        stage('Push Docker Images') {
            when {
                changeset 'src/services/**'
            }
            steps {
                script {
                    echo "Đăng nhập Docker Hub và đẩy các image..."
                    // Sử dụng docker.withRegistry để đăng nhập
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
                        // Push từng image (tag = IMAGE_TAG)
                        sh "docker push ${TTS_IMAGE_NAME}:${IMAGE_TAG}"
                        sh "docker push ${STT_IMAGE_NAME}:${IMAGE_TAG}"
                        sh "docker push ${CHATBOT_IMAGE_NAME}:${IMAGE_TAG}"

                        // Push tag latest
                        sh "docker push ${TTS_IMAGE_NAME}:latest"
                        sh "docker push ${STT_IMAGE_NAME}:latest"
                        sh "docker push ${CHATBOT_IMAGE_NAME}:latest"
                    }

                    echo "Push ảnh Docker hoàn tất."

                }
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo "Triển khai ứng dụng phiên bản: ${IMAGE_TAG} (placeholder)"
                // (k8s apply, ssh deploy, docker stack deploy, v.v.)
            }
        }
    }

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
