/**
 * Cấu hình ứng dụng Chat
 * Bạn có thể thay đổi các cài đặt này để tùy chỉnh ứng dụng
 */

window.CHAT_CONFIG = {
    // Cấu hình RAG bot API
    ollama: {
        // Bật/tắt sử dụng bot API (true/false)
        enabled: true,
        
        // URL API của chatbot service
        apiUrl: 'http://localhost:4096/answer/',
        
        // Tên model hiển thị trong giao diện
        model: 'RAG Bot'
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

