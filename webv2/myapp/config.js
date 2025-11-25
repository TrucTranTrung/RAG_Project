/**
 * Cấu hình ứng dụng Chat
 * Bạn có thể thay đổi các cài đặt này để tùy chỉnh ứng dụng
 */

window.CHAT_CONFIG = {
    // Cấu hình Ollama AI
    ollama: {
        // Bật/tắt sử dụng Ollama (true/false)
        enabled: true,

        // URL API của Ollama (mặc định: http://localhost:11434)
        apiUrl: 'http://localhost:4096/answer/',

        // Tên model Ollama bạn muốn sử dụng
        // Các model phổ biến: llama3.2, mistral, phi3, gemma2
        model: 'Gemini'
    },

    upload: {
        apiUrl: 'http://localhost:9001/upload'
    },

    // Cấu hình giao diện
    ui: {
        // Độ trễ tối thiểu khi hiển thị phản hồi (ms) - giảm để nhanh hơn
        typingDelayMin: 100,

        // Độ trễ tối đa khi hiển thị phản hồi (ms) - giảm để nhanh hơn
        typingDelayMax: 500,

        // Chiều cao tối đa của ô nhập liệu (px)
        maxTextareaHeight: 200,

        // Độ dài tối đa của tiêu đề cuộc trò chuyện
        titleMaxLength: 50
    }
};

