// Chat app main controller
(function() {
    'use strict';
    const STORAGE_KEYS = {
        CHATS: 'chatApp_conversations',
        MESSAGES: 'chatApp_messages',
        USER: 'chatApp_user',
        USERS: 'chatApp_users',
        THEME: 'chatApp_theme'
    };

    function formatDuration(seconds) {
        if (!seconds && seconds !== 0) return '';
        const total = Math.round(seconds);
        const mins = Math.floor(total / 60);
        const secs = total % 60;
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    function getConfig() {
        if (window.CHAT_CONFIG) {
            return {
                MAX_TEXTAREA_HEIGHT: window.CHAT_CONFIG.ui?.maxTextareaHeight || 200,
                TYPING_DELAY_MIN: window.CHAT_CONFIG.ui?.typingDelayMin || 800,
                TYPING_DELAY_MAX: window.CHAT_CONFIG.ui?.typingDelayMax || 2000,
                TITLE_MAX_LENGTH: window.CHAT_CONFIG.ui?.titleMaxLength || 50,
                OLLAMA_API_URL: window.CHAT_CONFIG.ollama?.apiUrl || 'http://localhost:4096/answer',
                OLLAMA_MODEL: window.CHAT_CONFIG.ollama?.model || 'gemini',
                USE_OLLAMA: window.CHAT_CONFIG.ollama?.enabled || true
            };
        }
        return {
            MAX_TEXTAREA_HEIGHT: 200,
            TYPING_DELAY_MIN: 800,
            TYPING_DELAY_MAX: 2000,
            TITLE_MAX_LENGTH: 50,
            OLLAMA_API_URL: 'http://localhost:4096/answer',
            OLLAMA_MODEL: 'gemini',
            USE_OLLAMA: true
        };
    };

    const CONFIG = getConfig();
    const MESSAGE_ROLES = {
        USER: 'user',
        ASSISTANT: 'assistant'
    };

    // App state
    const appState = {
        activeConversationId: null,
        conversations: [],
        messages: {},
        currentUser: null,
        isGuest: false
    };

    // DOM elements
    const domRefs = {
        authModal: null,
        appContainer: null,
        chatContainer: null,
        welcomeScreen: null,
        messageInput: null,
        sendButton: null,
        newChatButton: null,
        clearHistoryButton: null,
        chatHistoryContainer: null,
        sidebar: null,
        sidebarOverlay: null,
        mobileMenuButton: null,
        deleteMenu: null,
        deleteAllButton: null,
        deleteAllHistoryBtn: null,
        confirmModal: null,
        confirmModalOverlay: null,
        confirmModalContent: null,
        confirmCancelBtn: null,
        confirmConfirmBtn: null,
        aiStatus: null,
        loginTab: null,
        registerTab: null,
        loginForm: null,
        registerForm: null,
        guestBtn: null,
        userProfile: null,
        userMenuBtn: null,
        userMenuDropdown: null,
        userName: null,
        userAvatar: null,
        userStatus: null,
        headerUserName: null,
        headerUserAvatar: null,
        profileBtn: null,
        logoutBtn: null,
        themeToggleBtn: null,
        imageBtn: null,
        imageInput: null,
        recordBtn: null,
        stopRecordBtn: null,
        cancelRecordBtn: null,
        recordingStatus: null,
        recordingTime: null,
        inputPreview: null,
        previewItems: null
    };

    function initDOM() {
        domRefs.authModal = document.getElementById('authModal');
        domRefs.appContainer = document.getElementById('appContainer');
        domRefs.chatContainer = document.getElementById('chatContainer');
        domRefs.welcomeScreen = document.getElementById('welcomeScreen');
        domRefs.messageInput = document.getElementById('messageInput');
        domRefs.sendButton = document.getElementById('sendBtn');
        domRefs.newChatButton = document.getElementById('newChatBtn');
        domRefs.clearHistoryButton = document.getElementById('clearHistoryBtn');
        domRefs.chatHistoryContainer = document.getElementById('chatHistory');
        domRefs.sidebar = document.getElementById('sidebar');
        domRefs.sidebarOverlay = document.getElementById('sidebarOverlay');
        domRefs.mobileMenuButton = document.getElementById('mobileMenuBtn');
        domRefs.deleteMenu = document.getElementById('deleteMenu');
        domRefs.deleteAllButton = document.getElementById('deleteAllBtn');
        domRefs.deleteAllHistoryBtn = document.getElementById('deleteAllHistoryBtn');
        domRefs.confirmModal = document.getElementById('confirmModal');
        domRefs.confirmModalOverlay = document.getElementById('confirmModalOverlay');
        domRefs.confirmModalContent = document.getElementById('confirmModalContent');
        domRefs.confirmCancelBtn = document.getElementById('confirmCancelBtn');
        domRefs.confirmConfirmBtn = document.getElementById('confirmConfirmBtn');
        domRefs.aiStatus = document.getElementById('aiStatus');
        domRefs.loginTab = document.getElementById('loginTab');
        domRefs.registerTab = document.getElementById('registerTab');
        domRefs.loginForm = document.getElementById('loginForm');
        domRefs.registerForm = document.getElementById('registerForm');
        domRefs.guestBtn = document.getElementById('guestBtn');
        domRefs.userProfile = document.getElementById('userProfile');
        domRefs.userMenuBtn = document.getElementById('userMenuBtn');
        domRefs.userMenuDropdown = document.getElementById('userMenuDropdown');
        domRefs.userName = document.getElementById('userName');
        domRefs.userAvatar = document.getElementById('userAvatar');
        domRefs.userStatus = document.getElementById('userStatus');
        domRefs.headerUserName = document.getElementById('headerUserName');
        domRefs.headerUserAvatar = document.getElementById('headerUserAvatar');
        domRefs.profileBtn = document.getElementById('profileBtn');
        domRefs.logoutBtn = document.getElementById('logoutBtn');
        domRefs.themeToggleBtn = document.getElementById('themeToggleBtn');
        domRefs.imageBtn = document.getElementById('imageBtn');
        domRefs.imageInput = document.getElementById('imageInput');
        domRefs.recordBtn = document.getElementById('recordBtn');
        domRefs.stopRecordBtn = document.getElementById('stopRecordBtn');
        domRefs.cancelRecordBtn = document.getElementById('cancelRecordBtn');
        domRefs.recordingStatus = document.getElementById('recordingStatus');
        domRefs.recordingTime = document.getElementById('recordingTime');
        domRefs.inputPreview = document.getElementById('inputPreview');
        domRefs.previewItems = document.getElementById('previewItems');
    }

    // Local storage helper
    const storage = {
        save: function() {
            try {
                localStorage.setItem(STORAGE_KEYS.CHATS, JSON.stringify(appState.conversations));
                localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(appState.messages));
                if (appState.currentUser) {
                    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(appState.currentUser));
                }
            } catch (error) {
                console.warn('Failed to save to localStorage:', error);
            }
        },

        load: function() {
            try {
                const savedConversations = localStorage.getItem(STORAGE_KEYS.CHATS);
                const savedMessages = localStorage.getItem(STORAGE_KEYS.MESSAGES);
                const savedUser = localStorage.getItem(STORAGE_KEYS.USER);

                if (savedConversations) {
                    appState.conversations = JSON.parse(savedConversations);
                }

                if (savedMessages) {
                    appState.messages = JSON.parse(savedMessages);
                }

                if (savedUser) {
                    appState.currentUser = JSON.parse(savedUser);
                    appState.isGuest = appState.currentUser.isGuest || false;
                }
            } catch (error) {
                console.warn('Failed to load from localStorage:', error);
                appState.conversations = [];
                appState.messages = {};
            }
        },

        saveUsers: function(users) {
            try {
                localStorage.setItem(STORAGE_KEYS.USERS, JSON.stringify(users));
            } catch (error) {
                console.warn('Failed to save users:', error);
            }
        },

        loadUsers: function() {
            try {
                const savedUsers = localStorage.getItem(STORAGE_KEYS.USERS);
                return savedUsers ? JSON.parse(savedUsers) : {};
            } catch (error) {
                console.warn('Failed to load users:', error);
                return {};
            }
        }
    };

    // UI helpers
    const ui = {
        adjustTextareaHeight(textarea) {
            if (!textarea) return;
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, CONFIG.MAX_TEXTAREA_HEIGHT);
            textarea.style.height = newHeight + 'px';
        },

        scrollToBottom(container) {
            if (!container) return;
            requestAnimationFrame(() => {
                container.scrollTop = container.scrollHeight;
            });
        },

        sanitizeText(text) {
            if (typeof text !== 'string') return '';
            return text.trim();
        },

        truncateText(text, maxLength) {
            if (!text || text.length <= maxLength) return text;
            return text.substring(0, maxLength) + '...';
        }
    };

    // Message rendering
    const msgRenderer = {
        createMessageElement: function(role, content, attachments) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.setAttribute('role', 'listitem');

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === MESSAGE_ROLES.USER ? 'B' : 'AI';
            avatar.setAttribute('aria-label', role === MESSAGE_ROLES.USER ? 'Người dùng' : 'Trợ lý AI');

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // 处理附件（图片和音频）
            if (attachments) {
                const attachmentsDiv = document.createElement('div');
                attachmentsDiv.className = 'message-attachments';
                
                // 处理图片
                if (attachments.images && attachments.images.length > 0) {
                    attachments.images.forEach(imageSrc => {
                        const imgWrapper = document.createElement('div');
                        imgWrapper.className = 'message-image-wrapper';
                        const img = document.createElement('img');
                        img.src = imageSrc;
                        img.className = 'message-image';
                        img.alt = 'Hình ảnh đã gửi';
                        img.loading = 'lazy';
                        imgWrapper.appendChild(img);
                        attachmentsDiv.appendChild(imgWrapper);
                    });
                }
                
                // 处理音频
                if (attachments.audio) {
                    const audioWrapper = document.createElement('div');
                    audioWrapper.className = 'message-audio-wrapper';

                    const card = attachmentMgr.createAudioCard({
                        src: attachments.audio,
                        waveform: attachments.audioWaveform,
                        duration: attachments.audioDuration
                    });
                    audioWrapper.appendChild(card.container);

                    const actionRow = document.createElement('div');
                    actionRow.className = 'message-audio-actions';
                    const downloadLink = document.createElement('a');
                    downloadLink.href = attachments.audio;
                    downloadLink.target = '_blank';
                    downloadLink.rel = 'noopener';
                    downloadLink.textContent = 'Tải xuống';
                    downloadLink.className = 'message-audio-download';
                    downloadLink.download = `audio_${Date.now()}.mp3`;
                    actionRow.appendChild(downloadLink);

                    audioWrapper.appendChild(actionRow);
                    attachmentsDiv.appendChild(audioWrapper);
                }
                
                contentDiv.appendChild(attachmentsDiv);
            }
            
            // 处理文本内容
            if (content) {
                const textDiv = document.createElement('div');
                textDiv.className = 'message-text';
                textDiv.textContent = content;
                contentDiv.appendChild(textDiv);
            }

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);

            return messageDiv;
        },

        renderMessage(role, content, attachments) {
            const messageElement = this.createMessageElement(role, content, attachments);
            domRefs.chatContainer.appendChild(messageElement);
            ui.scrollToBottom(domRefs.chatContainer);
        },

        renderMessageWithTyping(role, messageId) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.id = messageId;
            messageDiv.setAttribute('role', 'listitem');

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === MESSAGE_ROLES.USER ? 'B' : 'AI';
            avatar.setAttribute('aria-label', role === MESSAGE_ROLES.USER ? 'Người dùng' : 'Trợ lý AI');

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = '';

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            domRefs.chatContainer.appendChild(messageDiv);
            ui.scrollToBottom(domRefs.chatContainer);
            return contentDiv;
        },

        updateMessageContent(messageId, content) {
            const messageElement = document.getElementById(messageId);
            if (messageElement) {
                const contentDiv = messageElement.querySelector('.message-content');
                if (contentDiv) {
                    contentDiv.textContent = content;
                    ui.scrollToBottom(domRefs.chatContainer);
                }
            }
        },

        typeMessageCharacterByCharacter(messageId, fullText, currentIndex, speed) {
            if (currentIndex >= fullText.length) {
                return;
            }

            const messageElement = document.getElementById(messageId);
            if (!messageElement) return;

            const contentDiv = messageElement.querySelector('.message-content');
            if (!contentDiv) return;

            // Thêm từng ký tự
            contentDiv.textContent = fullText.substring(0, currentIndex + 1);
            ui.scrollToBottom(domRefs.chatContainer);

            // Tốc độ gõ: nhanh hơn cho ký tự thường, chậm hơn cho dấu
            const char = fullText[currentIndex];
            let nextDelay = speed;
            
            // Nếu là dấu hoặc khoảng trắng, chậm hơn một chút
            if (/[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]/.test(char)) {
                nextDelay = speed * 1.5;
            }

            setTimeout(() => {
                this.typeMessageCharacterByCharacter(messageId, fullText, currentIndex + 1, speed);
            }, nextDelay);
        },

        createTypingIndicator() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            const indicatorId = 'typing_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            messageDiv.id = indicatorId;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = 'AI';

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.innerHTML = '<span></span><span></span><span></span>';

            contentDiv.appendChild(typingDiv);
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            domRefs.chatContainer.appendChild(messageDiv);

            ui.scrollToBottom(domRefs.chatContainer);
            return indicatorId;
        },

        removeTypingIndicator(indicatorId) {
            const indicator = document.getElementById(indicatorId);
            if (indicator) {
                indicator.remove();
            }
        }
    };

    // Mobile menu
    const mobileMenu = {
        open: function() {
            if (domRefs.sidebar) {
                domRefs.sidebar.classList.add('open');
            }
            if (domRefs.sidebarOverlay) {
                domRefs.sidebarOverlay.classList.add('active');
            }
            document.body.style.overflow = 'hidden';
        },

        close() {
            if (domRefs.sidebar) {
                domRefs.sidebar.classList.remove('open');
            }
            if (domRefs.sidebarOverlay) {
                domRefs.sidebarOverlay.classList.remove('active');
            }
            document.body.style.overflow = '';
        },

        toggle() {
            if (domRefs.sidebar && domRefs.sidebar.classList.contains('open')) {
                this.close();
            } else {
                this.open();
            }
        },

        isMobile() {
            return window.innerWidth <= 768;
        }
    };

    // AI integration
    const ai = {
        generateResponse: async function(userMessage, conversationHistory, onStream) {
            console.log('[AI Service] Đang sử dụng Gemini AI');
            return await this.generateWithOllama(userMessage, conversationHistory, onStream);
        },

        async generateWithOllama(userMessage, conversationHistory, onStream) {
            try {
                console.log('[Bot] Đang gửi request đến bot...');
                const startTime = Date.now();

                // Gửi trực tiếp userMessage (không gộp history) vì bot chấp nhận string
                const formData = new FormData();
                formData.append('question', userMessage);
                const response = await fetch(`${CONFIG.OLLAMA_API_URL}`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Bot API error: ${response.status} - ${errorText}`);
                }

                // Backend trả JSON (text hoặc audio_base64)
                const data = await response.json();

                // Nếu backend trả audio_base64, trả về base64 (caller xử lý)
                if (data.audio_base64) {
                    const elapsedTime = Date.now() - startTime;
                    console.log(`[Bot] Received audio (base64) — elapsed ${elapsedTime}ms`);
                    return data.audio_base64;
                }

                // Nếu backend trả text, lấy trường content (hoặc fallback)
                let fullResponse = '';
                if (data.content) {
                    fullResponse = data.content;
                } else if (typeof data === 'string') {
                    fullResponse = data;
                } else if (data.type === 'text' && data.content) {
                    fullResponse = data.content;
                } else {
                    fullResponse = '';
                }

                // Stream character-by-character nếu có callback onStream
                if (onStream && typeof onStream === 'function') {
                    for (let i = 0; i < fullResponse.length; i++) {
                        onStream(fullResponse[i]);
                        // không bắt buộc delay — để 0 để nhanh, chỉnh nếu cần hiệu ứng typewriter
                        await new Promise(r => setTimeout(r, 0));
                    }
                }

                const elapsedTime = Date.now() - startTime;
                console.log(`[Bot] Hoàn thành sau ${elapsedTime}ms`);
                return fullResponse.trim();

            } catch (error) {
                console.error('[Bot] Lỗi:', error);
                console.warn('[Bot] Chuyển sang template mode');
                return `${error}`;
            }
        },


        // generateWithTemplate: function(userMessage) {
        //     return responseGenerator.generateResponse(userMessage);
        // }
    };

    // Response Generator
    // const responseGenerator = {
    //     generateResponse: function(userMessage) {
    //         // const normalizedMessage = userMessage.toLowerCase();
    //         return this.getAnalysisResponse();
    //     },

    //     getAnalysisResponse: function() {
    //         return `Để phân tích vấn đề này, tôi sẽ xem xét:

    //                 • Nguyên nhân gốc rễ
    //                 • Các yếu tố ảnh hưởng
    //                 • Giải pháp khả thi
    //                 • Đánh giá rủi ro và lợi ích

    //                 Bạn có thể cung cấp thêm thông tin chi tiết về vấn đề không?`;
    //     },
    // };

    // Chat management
    const chatMgr = {
        createNew: function() {
            appState.activeConversationId = null;
            domRefs.chatContainer.innerHTML = '';
            domRefs.welcomeScreen.style.display = 'flex';
            domRefs.messageInput.value = '';
            ui.adjustTextareaHeight(domRefs.messageInput);
            domRefs.sendButton.disabled = true;
            this.updateHistoryView();
            domRefs.messageInput.focus();
            
            if (mobileMenu.isMobile()) {
                mobileMenu.close();
            }
        },

        initializeNew(firstMessage) {
            const conversationId = 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            const title = ui.truncateText(firstMessage, CONFIG.TITLE_MAX_LENGTH);

            appState.activeConversationId = conversationId;
            appState.conversations.push({
                id: conversationId,
                title: title,
                timestamp: Date.now()
            });
            appState.messages[conversationId] = [];

            domRefs.welcomeScreen.style.display = 'none';
            this.updateHistoryView();
        },

        load(conversationId) {
            if (!appState.messages[conversationId]) {
                console.warn('Conversation not found:', conversationId);
                return;
            }

            appState.activeConversationId = conversationId;
            domRefs.chatContainer.innerHTML = '';
            domRefs.welcomeScreen.style.display = 'none';

            const messages = appState.messages[conversationId];
            messages.forEach(msg => {
                msgRenderer.renderMessage(msg.role, msg.content, msg.attachments);
            });

            this.updateHistoryView();
            domRefs.messageInput.focus();
            
            if (mobileMenu.isMobile()) {
                mobileMenu.close();
            }
        },

        updateHistoryView() {
            if (!domRefs.chatHistoryContainer) return;

            domRefs.chatHistoryContainer.innerHTML = '';

            const sortedConversations = [...appState.conversations]
                .sort((a, b) => b.timestamp - a.timestamp);

            sortedConversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'chat-history-item';
                item.setAttribute('role', 'listitem');

                if (conv.id === appState.activeConversationId) {
                    item.classList.add('active');
                }

                item.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                        <path d="M2 3H14M2 8H14M2 13H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <span>${this.escapeHtml(conv.title)}</span>
                    <button class="chat-history-item-delete" type="button" aria-label="Xóa cuộc trò chuyện này" data-conv-id="${conv.id}">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                            <path d="M3.5 3.5H10.5M9.5 3.5V11.5C9.5 11.9142 9.31571 12.2893 9.0375 12.5375C8.75929 12.7857 8.41421 12.9 8 12.9H6C5.58579 12.9 5.24071 12.7857 4.9625 12.5375C4.68429 12.2893 4.5 11.9142 4.5 11.5V3.5M5.75 3.5V2.625C5.75 2.21079 5.93429 1.83571 6.2125 1.5875C6.49071 1.33929 6.83579 1.225 7.25 1.225C7.66421 1.225 8.00929 1.33929 8.2875 1.5875C8.56571 1.83571 8.75 2.21079 8.75 2.625V3.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                `;

                const deleteBtn = item.querySelector('.chat-history-item-delete');
                if (deleteBtn) {
                    deleteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        this.deleteConversation(conv.id);
                    });
                }

                item.addEventListener('click', (e) => {
                    if (deleteBtn && (e.target === deleteBtn || deleteBtn.contains(e.target))) {
                        return;
                    }
                    this.load(conv.id);
                    if (mobileMenu.isMobile()) {
                        mobileMenu.close();
                    }
                });

                domRefs.chatHistoryContainer.appendChild(item);
            });
        },

        deleteConversation(conversationId) {
            const conversation = appState.conversations.find(c => c.id === conversationId);
            const title = conversation ? conversation.title : '';
            confirmDialog.showDeleteSingle(conversationId, title);
        },

        deleteConversationDirect(conversationId) {
            appState.conversations = appState.conversations.filter(c => c.id !== conversationId);
            delete appState.messages[conversationId];

            if (appState.activeConversationId === conversationId) {
                if (appState.conversations.length > 0) {
                    this.load(appState.conversations[0].id);
                } else {
                    this.createNew();
                }
            } else {
                this.updateHistoryView();
            }

            storage.save();
        },

        clearAll() {
            appState.conversations = [];
            appState.messages = {};
            appState.activeConversationId = null;
            this.createNew();
            storage.save();
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    };

    const audioController = {
        currentAudio: null,
        currentButton: null,

        attach(audioEl, button) {
            if (!audioEl || !button) return;

            button.addEventListener('click', () => {
                if (audioEl.paused) {
                    this.play(audioEl, button);
                } else {
                    audioEl.pause();
                }
            });

            audioEl.addEventListener('play', () => {
                button.classList.add('playing');
            });

            audioEl.addEventListener('pause', () => {
                button.classList.remove('playing');
            });

            audioEl.addEventListener('ended', () => {
                button.classList.remove('playing');
                audioEl.currentTime = 0;
                if (this.currentAudio === audioEl) {
                    this.currentAudio = null;
                    this.currentButton = null;
                }
            });
        },

        play(audioEl, button) {
            if (this.currentAudio && this.currentAudio !== audioEl) {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
                if (this.currentButton) {
                    this.currentButton.classList.remove('playing');
                }
            }
            this.currentAudio = audioEl;
            this.currentButton = button;
            const playPromise = audioEl.play();
            if (playPromise?.catch) {
                playPromise.catch((err) => {
                    console.warn('Audio play failed:', err);
                    button.classList.remove('playing');
                });
            }
        }
    };

    // Attachment manager
    const attachmentMgr = {
        images: [],
        audio: null, // { blob, preview, waveform, duration }
        recording: null,
        mediaRecorder: null,
        recordingStream: null,
        recordingTimer: null,
        recordingStartTime: null,

        addImage: function(file) {
            const preview = URL.createObjectURL(file);
            this.images.push({
                file,
                preview
            });
            this.updatePreview();
        },

        removeImage: function(index) {
            const removed = this.images.splice(index, 1);
            if (removed[0]?.preview) {
                URL.revokeObjectURL(removed[0].preview);
            }
            this.updatePreview();
        },

        startRecording: function() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    this.mediaRecorder = new MediaRecorder(stream);
                    this.recordingStream = stream; // 保存stream引用
                    const chunks = [];

                    this.mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) {
                            chunks.push(e.data);
                        }
                    };

                    this.mediaRecorder.onstop = () => {
                        const blob = new Blob(chunks, { type: 'audio/webm' });
                        if (this.audio?.preview) {
                            URL.revokeObjectURL(this.audio.preview);
                        }
                        this.audio = {
                            blob,
                            preview: URL.createObjectURL(blob),
                            waveform: null,
                            duration: null
                        };
                        this.generateWaveform(blob).then((result) => {
                            if (this.audio) {
                                this.audio.waveform = result.waveform;
                                this.audio.duration = result.duration;
                                this.updatePreview();
                            }
                        }).catch(() => {
                            this.updatePreview();
                        });
                        if (this.recordingStream) {
                            this.recordingStream.getTracks().forEach(track => track.stop());
                            this.recordingStream = null;
                        }
                    };

                    this.mediaRecorder.start();
                    this.recordingStartTime = Date.now();
                    this.startTimer();
                    domRefs.recordingStatus.style.display = 'flex';
                    domRefs.recordBtn.classList.add('recording');
                })
                .catch(err => {
                    console.error('Lỗi khi bắt đầu ghi âm:', err);
                    alert('Không thể truy cập microphone. Vui lòng kiểm tra quyền truy cập.');
                });
        },

        stopRecording: function() {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
                this.stopTimer();
                domRefs.recordingStatus.style.display = 'none';
                domRefs.recordBtn.classList.remove('recording');
            }
        },

        cancelRecording: function() {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
            }
            if (this.recordingStream) {
                this.recordingStream.getTracks().forEach(track => track.stop());
                this.recordingStream = null;
            }
            this.audio = null;
            this.stopTimer();
            domRefs.recordingStatus.style.display = 'none';
            domRefs.recordBtn.classList.remove('recording');
            this.updatePreview();
        },

        startTimer: function() {
            this.recordingTimer = setInterval(() => {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                domRefs.recordingTime.textContent = 
                    `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            }, 1000);
        },

        stopTimer: function() {
            if (this.recordingTimer) {
                clearInterval(this.recordingTimer);
                this.recordingTimer = null;
            }
        },

        updatePreview: function() {
            domRefs.previewItems.innerHTML = '';
            
            if (this.images.length === 0 && !this.audio) {
                domRefs.inputPreview.style.display = 'none';
                return;
            }

            domRefs.inputPreview.style.display = 'block';

            // 显示图片预览
            this.images.forEach((imageObj, index) => {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item preview-image';
                const img = document.createElement('img');
                img.src = imageObj.preview;
                img.alt = 'Preview';
                const removeBtn = document.createElement('button');
                removeBtn.className = 'preview-remove';
                removeBtn.innerHTML = '×';
                removeBtn.onclick = () => this.removeImage(index);
                previewItem.appendChild(img);
                previewItem.appendChild(removeBtn);
                domRefs.previewItems.appendChild(previewItem);
            });

            // 显示音频预览
            if (this.audio) {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item preview-audio';

                const card = this.createAudioCard({
                    src: this.audio.preview,
                    waveform: this.audio.waveform,
                    duration: this.audio.duration
                });
                previewItem.appendChild(card.container);

                const removeBtn = document.createElement('button');
                removeBtn.className = 'preview-remove';
                removeBtn.innerHTML = '×';
                removeBtn.onclick = () => {
                    if (this.audio?.preview) {
                        URL.revokeObjectURL(this.audio.preview);
                    }
                    this.audio = null;
                    this.updatePreview();
                };
                previewItem.appendChild(removeBtn);
                domRefs.previewItems.appendChild(previewItem);
            }

            // 更新发送按钮状态
            const hasContent = ui.sanitizeText(domRefs.messageInput.value).length > 0 || 
                              this.images.length > 0 || 
                              this.audio !== null;
            domRefs.sendButton.disabled = !hasContent;
        },

        clear: function() {
            this.images.forEach((img) => {
                if (img.preview) {
                    URL.revokeObjectURL(img.preview);
                }
            });
            this.images = [];
            if (this.audio?.preview) {
                URL.revokeObjectURL(this.audio.preview);
            }
            this.audio = null;
            if (this.recording) {
                URL.revokeObjectURL(this.recording);
                this.recording = null;
            }
            this.updatePreview();
        },

        createAudioCard: function({ src, waveform, duration }) {
            const container = document.createElement('div');
            container.className = 'audio-card';

            const playButton = document.createElement('button');
            playButton.className = 'audio-play-button';
            playButton.setAttribute('aria-label', 'Phát âm thanh');
            container.appendChild(playButton);

            const waveformWrapper = document.createElement('div');
            waveformWrapper.className = 'audio-waveform-wrapper';
            const waveformCanvas = document.createElement('canvas');
            waveformCanvas.width = 220;
            waveformCanvas.height = 48;
            waveformCanvas.className = 'waveform-canvas';
            if (waveform) {
                this.drawWaveform(waveformCanvas, waveform);
            }
            waveformWrapper.appendChild(waveformCanvas);
            const durationLabel = document.createElement('span');
            durationLabel.className = 'audio-duration-label';
            durationLabel.textContent = duration ? formatDuration(duration) : '--:--';
            waveformWrapper.appendChild(durationLabel);
            container.appendChild(waveformWrapper);

            const audioEl = document.createElement('audio');
            audioEl.src = src;
            audioEl.preload = 'auto';
            audioEl.className = 'audio-element-hidden';
            container.appendChild(audioEl);

            audioController.attach(audioEl, playButton);

            return { container, playButton, audioEl };
        },

        async generateWaveform(blob) {
            if (!window.AudioContext && !window.webkitAudioContext) return { waveform: null, duration: null };
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const audioCtx = new AudioCtx();
            const arrayBuffer = await blob.arrayBuffer();
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const rawData = audioBuffer.getChannelData(0);
            const samples = 80;
            const blockSize = Math.floor(rawData.length / samples);
            const waveform = [];
            for (let i = 0; i < samples; i++) {
                let sum = 0;
                for (let j = 0; j < blockSize; j++) {
                    sum += Math.abs(rawData[i * blockSize + j]);
                }
                waveform.push(sum / blockSize);
            }
            audioCtx.close();
            return {
                waveform,
                duration: audioBuffer.duration
            };
        },

        drawWaveform(canvas, data) {
            if (!canvas || !data) return;
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = 'rgba(255,255,255,0.1)';
            ctx.fillRect(0, 0, width, height);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const step = width / data.length;
            for (let i = 0; i < data.length; i++) {
                const x = i * step;
                const value = data[i];
                const y = height / 2;
                const barHeight = value * height;
                ctx.moveTo(x, y - barHeight / 2);
                ctx.lineTo(x, y + barHeight / 2);
            }
            ctx.stroke();
        }
    };

    // Uploader
    const uploader = {
        async uploadImage(file) {
            const formData = new FormData();
            formData.append('type', 'image');
            formData.append('file', file, file.name || `image_${Date.now()}.jpg`);

            const response = await fetch('upload.php', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Không thể tải ảnh lên');
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.message || 'Tải ảnh thất bại');
            }
            return data.payload;
        },

        async uploadAudio(blob) {
            const formData = new FormData();
            formData.append('type', 'audio');
            const audioFile = new File([blob], `audio_${Date.now()}.webm`, { type: 'audio/webm' });
            formData.append('file', audioFile);

            const response = await fetch('upload.php', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Không thể tải âm thanh lên');
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.message || 'Tải âm thanh thất bại');
            }
            return data.payload;
        },

        async uploadAll(state) {
            const results = {
                images: [],
                audio: null
            };

            if (state.images.length > 0) {
                const uploads = state.images.map((img) => this.uploadImage(img.file));
                results.images = await Promise.all(uploads);
            }

            if (state.audio?.blob) {
                results.audio = await this.uploadAudio(state.audio.blob);
            }

            return results;
        }
    };

    // Message handler
    const msgHandler = {
        send: async function() {
            const messageText = ui.sanitizeText(domRefs.messageInput.value);
            const hasAttachments = attachmentMgr.images.length > 0 || attachmentMgr.audio !== null;
            
            if (!messageText && !hasAttachments) return;

            if (!appState.activeConversationId) {
                chatMgr.initializeNew(messageText || 'Tin nhắn có đính kèm');
            }

            domRefs.sendButton.disabled = true;
            domRefs.sendButton.classList.add('is-uploading');

            let uploadedAttachments = { images: [], audio: null };
            if (hasAttachments) {
                try {
                    uploadedAttachments = await uploader.uploadAll({
                        images: attachmentMgr.images,
                        audio: attachmentMgr.audio
                    });
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('Không thể tải tệp đính kèm. Vui lòng thử lại.');
                    domRefs.sendButton.disabled = false;
                    domRefs.sendButton.classList.remove('is-uploading');
                    return;
                }
            }

            const attachments = {};
            if (uploadedAttachments.images.length > 0) {
                attachments.images = uploadedAttachments.images.map(item => item.url);
            }
            if (uploadedAttachments.audio) {
                attachments.audio = uploadedAttachments.audio.url;
                attachments.audioDuration = uploadedAttachments.audio.duration;
            } else if (attachmentMgr.audio?.duration) {
                attachments.audioDuration = attachmentMgr.audio.duration;
            }
            if (attachmentMgr.audio?.waveform) {
                attachments.audioWaveform = attachmentMgr.audio.waveform;
            }

            msgRenderer.renderMessage(
                MESSAGE_ROLES.USER,
                messageText,
                Object.keys(attachments).length > 0 ? attachments : null
            );
            
            appState.messages[appState.activeConversationId].push({
                role: MESSAGE_ROLES.USER,
                content: messageText,
                attachments: Object.keys(attachments).length > 0 ? attachments : undefined
            });

            domRefs.messageInput.value = '';
            attachmentMgr.clear();
            ui.adjustTextareaHeight(domRefs.messageInput);
            domRefs.sendButton.disabled = true;
            domRefs.sendButton.classList.remove('is-uploading');

            const typingIndicatorId = msgRenderer.createTypingIndicator();

            const delay = CONFIG.USE_OLLAMA ? 
                Math.min(CONFIG.TYPING_DELAY_MIN, 200) :
                CONFIG.TYPING_DELAY_MIN + Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN);

            setTimeout(async () => {
                msgRenderer.removeTypingIndicator(typingIndicatorId);
                const conversationHistory = appState.messages[appState.activeConversationId] || [];
                const startTime = Date.now();
                
                const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                msgRenderer.renderMessageWithTyping(MESSAGE_ROLES.ASSISTANT, messageId);
                
                // Lấy response từ Ollama
                const response = await ai.generateResponse(messageText, conversationHistory);
                const totalTime = Date.now() - startTime;
                console.log(`[Message Handler] Tổng thời gian: ${totalTime}ms`);
                
                // Hiển thị từng ký tự giống như đang gõ telex
                const typingSpeed = 30; // milliseconds per character (có thể điều chỉnh)
                msgRenderer.typeMessageCharacterByCharacter(messageId, response, 0, typingSpeed);
                
                setTimeout(() => {
                    appState.messages[appState.activeConversationId].push({
                        role: MESSAGE_ROLES.ASSISTANT,
                        content: response
                    });
                    storage.save();
                }, response.length * typingSpeed + 100);
            }, delay);
        }
    };

    // Auth
    const auth = {
        showModal: function() {
            if (domRefs.authModal) {
                domRefs.authModal.classList.add('show');
            }
        },

        hideModal() {
            if (domRefs.authModal) {
                domRefs.authModal.classList.remove('show');
            }
        },

        switchTab(tab) {
            if (tab === 'login') {
                if (domRefs.loginTab) domRefs.loginTab.classList.add('active');
                if (domRefs.registerTab) domRefs.registerTab.classList.remove('active');
                if (domRefs.loginForm) domRefs.loginForm.classList.add('active');
                if (domRefs.registerForm) domRefs.registerForm.classList.remove('active');
            } else {
                if (domRefs.loginTab) domRefs.loginTab.classList.remove('active');
                if (domRefs.registerTab) domRefs.registerTab.classList.add('active');
                if (domRefs.loginForm) domRefs.loginForm.classList.remove('active');
                if (domRefs.registerForm) domRefs.registerForm.classList.add('active');
            }
        },

        login(email, password) {
            const users = storage.loadUsers();
            const user = users[email];

            if (!user) {
                throw new Error('Email không tồn tại');
            }

            if (user.password !== password) {
                throw new Error('Mật khẩu không đúng');
            }

            appState.currentUser = {
                email: user.email,
                name: user.name,
                isGuest: false
            };
            appState.isGuest = false;

            storage.save();
            this.updateUserUI();
            this.hideModal();
            if (domRefs.appContainer) {
                domRefs.appContainer.style.display = 'flex';
            }
        },

        register(name, email, password) {
            const users = storage.loadUsers();

            if (users[email]) {
                throw new Error('Email đã được sử dụng');
            }

            users[email] = {
                name: name,
                email: email,
                password: password,
                createdAt: Date.now()
            };

            storage.saveUsers(users);

            appState.currentUser = {
                email: email,
                name: name,
                isGuest: false
            };
            appState.isGuest = false;

            storage.save();
            this.updateUserUI();
            this.hideModal();
            if (domRefs.appContainer) {
                domRefs.appContainer.style.display = 'flex';
            }
        },

        guestLogin() {
            appState.currentUser = {
                name: 'Khách',
                email: 'guest_' + Date.now(),
                isGuest: true
            };
            appState.isGuest = true;

            storage.save();
            this.updateUserUI();
            this.hideModal();
            if (domRefs.appContainer) {
                domRefs.appContainer.style.display = 'flex';
            }
        },

        logout() {
            appState.currentUser = null;
            appState.isGuest = false;
            localStorage.removeItem(STORAGE_KEYS.USER);
            this.showModal();
            if (domRefs.appContainer) {
                domRefs.appContainer.style.display = 'none';
            }
        },

        updateUserUI() {
            if (!appState.currentUser) return;

            const name = appState.currentUser.name || 'Khách';
            const initial = name.charAt(0).toUpperCase();
            const status = appState.isGuest ? 'Miễn phí' : 'Đã đăng nhập';

            if (domRefs.userName) {
                domRefs.userName.textContent = name;
            }
            if (domRefs.userAvatar) {
                domRefs.userAvatar.textContent = initial;
            }
            if (domRefs.userStatus) {
                domRefs.userStatus.textContent = status;
            }
            if (domRefs.headerUserName) {
                domRefs.headerUserName.textContent = name;
            }
            if (domRefs.headerUserAvatar) {
                domRefs.headerUserAvatar.textContent = initial;
            }
        },

        checkAuth() {
            storage.load();
            if (appState.currentUser) {
                this.updateUserUI();
                this.hideModal();
                if (domRefs.appContainer) {
                    domRefs.appContainer.style.display = 'flex';
                }
            } else {
                this.showModal();
                if (domRefs.appContainer) {
                    domRefs.appContainer.style.display = 'none';
                }
            }
        }
    };

    // Event setup
    function setupEvents() {
        domRefs.sendButton.addEventListener('click', () => {
            msgHandler.send();
        });

        domRefs.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                msgHandler.send();
            }
        });

        domRefs.messageInput.addEventListener('input', () => {
            ui.adjustTextareaHeight(domRefs.messageInput);
            attachmentMgr.updatePreview();
        });

        // 图片上传功能
        if (domRefs.imageBtn && domRefs.imageInput) {
            domRefs.imageBtn.addEventListener('click', () => {
                domRefs.imageInput.click();
            });

            domRefs.imageInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                files.forEach(file => {
                    if (file.type.startsWith('image/')) {
                        attachmentMgr.addImage(file);
                    }
                });
                e.target.value = ''; // 重置input以便可以再次选择同一文件
            });
        }

        // 录音功能
        if (domRefs.recordBtn) {
            domRefs.recordBtn.addEventListener('click', () => {
                if (attachmentMgr.mediaRecorder && attachmentMgr.mediaRecorder.state === 'recording') {
                    attachmentMgr.stopRecording();
                } else {
                    attachmentMgr.startRecording();
                }
            });
        }

        if (domRefs.stopRecordBtn) {
            domRefs.stopRecordBtn.addEventListener('click', () => {
                attachmentMgr.stopRecording();
            });
        }

        if (domRefs.cancelRecordBtn) {
            domRefs.cancelRecordBtn.addEventListener('click', () => {
                attachmentMgr.cancelRecording();
            });
        }

        domRefs.newChatButton.addEventListener('click', () => {
            chatMgr.createNew();
            if (mobileMenu.isMobile()) {
                mobileMenu.close();
            }
        });

            if (domRefs.deleteMenu) {
                domRefs.clearHistoryButton.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (domRefs.deleteMenu) {
                        domRefs.deleteMenu.classList.toggle('active');
                    }
                });

            }

        if (domRefs.deleteAllButton) {
            domRefs.deleteAllButton.addEventListener('click', () => {
                chatMgr.clearAll();
                if (domRefs.deleteMenu) {
                    domRefs.deleteMenu.classList.remove('active');
                }
                if (mobileMenu.isMobile()) {
                    mobileMenu.close();
                }
            });
        }

        // Auth
        if (domRefs.loginTab) {
            domRefs.loginTab.addEventListener('click', () => {
                auth.switchTab('login');
            });
        }

        if (domRefs.registerTab) {
            domRefs.registerTab.addEventListener('click', () => {
                auth.switchTab('register');
            });
        }

        if (domRefs.loginForm) {
            domRefs.loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('loginEmail').value;
                const password = document.getElementById('loginPassword').value;
                
                try {
                    auth.login(email, password);
                    domRefs.loginForm.reset();
                } catch (error) {
                    alert(error.message);
                }
            });
        }

        if (domRefs.registerForm) {
            domRefs.registerForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const name = document.getElementById('registerName').value;
                const email = document.getElementById('registerEmail').value;
                const password = document.getElementById('registerPassword').value;
                
                try {
                    auth.register(name, email, password);
                    domRefs.registerForm.reset();
                    auth.switchTab('login');
                } catch (error) {
                    alert(error.message);
                }
            });
        }

        if (domRefs.guestBtn) {
            domRefs.guestBtn.addEventListener('click', () => {
                auth.guestLogin();
            });
        }

        if (domRefs.userMenuBtn) {
            domRefs.userMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (domRefs.userProfile) {
                    domRefs.userProfile.classList.toggle('active');
                }
            });
        }

        if (domRefs.logoutBtn) {
            domRefs.logoutBtn.addEventListener('click', () => {
                auth.logout();
                if (domRefs.userProfile) {
                    domRefs.userProfile.classList.remove('active');
                }
            });
        }

        if (domRefs.profileBtn) {
            domRefs.profileBtn.addEventListener('click', () => {
                if (appState.currentUser) {
                    const info = appState.isGuest ? 
                        'Bạn đang sử dụng chế độ khách' : 
                        `Email: ${appState.currentUser.email}\nTên: ${appState.currentUser.name}`;
                    alert(info);
                }
                if (domRefs.userProfile) {
                    domRefs.userProfile.classList.remove('active');
                }
            });
        }

        if (domRefs.deleteAllHistoryBtn) {
            domRefs.deleteAllHistoryBtn.addEventListener('click', () => {
                confirmDialog.showDeleteAll();
                if (domRefs.userProfile) {
                    domRefs.userProfile.classList.remove('active');
                }
            });
        }

        if (domRefs.confirmCancelBtn) {
            domRefs.confirmCancelBtn.addEventListener('click', () => {
                confirmDialog.hide();
            });
        }

        if (domRefs.confirmConfirmBtn) {
            domRefs.confirmConfirmBtn.addEventListener('click', () => {
                confirmDialog.executeAction();
            });
        }

        if (domRefs.confirmModalOverlay) {
            domRefs.confirmModalOverlay.addEventListener('click', () => {
                confirmDialog.hide();
            });
        }

        if (domRefs.themeToggleBtn) {
            domRefs.themeToggleBtn.addEventListener('click', (e) => {
                theme.toggle(e);
            });
        }

        document.addEventListener('click', (e) => {
                if (domRefs.userProfile && !domRefs.userProfile.contains(e.target)) {
                    domRefs.userProfile.classList.remove('active');
                }
                if (domRefs.deleteMenu && !domRefs.deleteMenu.contains(e.target)) {
                    domRefs.deleteMenu.classList.remove('active');
                }
            });

            if (domRefs.mobileMenuButton) {
                domRefs.mobileMenuButton.addEventListener('click', () => {
                    mobileMenu.toggle();
                });
            }

            if (domRefs.sidebarOverlay) {
                domRefs.sidebarOverlay.addEventListener('click', () => {
                    mobileMenu.close();
                });
            }

            window.addEventListener('resize', () => {
                if (!mobileMenu.isMobile() && domRefs.sidebar) {
                    domRefs.sidebar.classList.remove('open');
                    if (domRefs.sidebarOverlay) {
                        domRefs.sidebarOverlay.classList.remove('active');
                    }
                    document.body.style.overflow = '';
                }
            });

            document.querySelectorAll('.suggestion-card').forEach(card => {
                card.addEventListener('click', () => {
                    const suggestion = card.getAttribute('data-suggestion');
                    if (suggestion) {
                        domRefs.messageInput.value = suggestion;
                        ui.adjustTextareaHeight(domRefs.messageInput);
                        domRefs.sendButton.disabled = false;
                        domRefs.messageInput.focus();
                    }
                });
            });
    }

    // Update AI Status
    function updateAIStatus() {
        if (domRefs.aiStatus) {
            if (CONFIG.USE_OLLAMA) {
                domRefs.aiStatus.textContent = `Đang dùng Ollama AI (${CONFIG.OLLAMA_MODEL})`;
                domRefs.aiStatus.style.color = 'var(--accent-color)';
            } else {
                domRefs.aiStatus.textContent = 'Đang dùng Template mode';
                domRefs.aiStatus.style.color = 'var(--text-secondary)';
            }
        }
    }

    // Confirmation dialog
    const confirmDialog = {
        pendingAction: null,
        pendingConversationId: null,

        show: function() {
            if (domRefs.confirmModal) {
                domRefs.confirmModal.classList.add('show');
                document.body.style.overflow = 'hidden';
            }
        },

        hide: function() {
            if (domRefs.confirmModal) {
                domRefs.confirmModal.classList.remove('show');
                document.body.style.overflow = '';
            }
            this.pendingAction = null;
            this.pendingConversationId = null;
        },

        showDeleteAll() {
            this.pendingAction = 'deleteAll';
            this.pendingConversationId = null;
            if (domRefs.confirmModalTitle) {
                domRefs.confirmModalTitle.textContent = 'Xóa lịch sử chat';
            }
            if (domRefs.confirmModalMessage) {
                domRefs.confirmModalMessage.textContent = 'Bạn có muốn xóa tất cả lịch sử chat không?';
            }
            this.show();
        },

        showDeleteSingle(conversationId, conversationTitle) {
            this.pendingAction = 'deleteSingle';
            this.pendingConversationId = conversationId;
            if (domRefs.confirmModalTitle) {
                domRefs.confirmModalTitle.textContent = 'Xóa cuộc trò chuyện';
            }
            if (domRefs.confirmModalMessage) {
                const title = uiUtils.truncateText(conversationTitle || 'cuộc trò chuyện này', 30);
                domRefs.confirmModalMessage.textContent = `Bạn có muốn xóa "${title}" không?`;
            }
            this.show();
        },

        executeAction() {
            if (this.pendingAction === 'deleteAll') {
                chatMgr.clearAll();
            } else if (this.pendingAction === 'deleteSingle' && this.pendingConversationId) {
                chatMgr.deleteConversationDirect(this.pendingConversationId);
            }
            this.hide();
        }
    };

    // Theme manager
    const theme = {
        current: 'dark',
        isTransitioning: false,

        init() {
            const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME);
            if (savedTheme === 'light' || savedTheme === 'dark') {
                this.current = savedTheme;
            } else {
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                this.current = prefersDark ? 'dark' : 'light';
            }
            this.apply();
        },

        toggle(event) {
            if (this.isTransitioning) return;
            
            const newTheme = this.current === 'dark' ? 'light' : 'dark';
            this.animateTransition(event, newTheme);
        },

        animateTransition(event, newTheme) {
            this.isTransitioning = true;
            const overlay = document.getElementById('themeTransitionOverlay');
            if (!overlay) {
                this.applyTheme(newTheme);
                return;
            }

            const button = event?.target?.closest('.theme-toggle-btn') || domRefs.themeToggleBtn;
            if (!button) {
                this.applyTheme(newTheme);
                return;
            }

            const rect = button.getBoundingClientRect();
            const x = rect.left + rect.width / 2;
            const y = rect.top + rect.height / 2;
            
            const maxDimension = Math.max(window.innerWidth, window.innerHeight);
            const size = Math.sqrt(maxDimension * maxDimension * 2) * 2.5;

            overlay.style.setProperty('--ripple-x', `${x}px`);
            overlay.style.setProperty('--ripple-y', `${y}px`);
            overlay.style.setProperty('--ripple-size', `${size}px`);
            overlay.setAttribute('data-theme', newTheme);
            
            requestAnimationFrame(() => {
                overlay.classList.add('active');
                
                setTimeout(() => {
                    this.current = newTheme;
                    this.apply();
                    localStorage.setItem(STORAGE_KEYS.THEME, this.current);
                }, 250);
                
                setTimeout(() => {
                    overlay.classList.remove('active');
                    overlay.style.width = '0';
                    overlay.style.height = '0';
                    this.isTransitioning = false;
                }, 900);
            });
        },

        applyTheme(newTheme) {
            this.current = newTheme;
            this.apply();
            localStorage.setItem(STORAGE_KEYS.THEME, this.current);
            this.isTransitioning = false;
        },

        apply() {
            document.documentElement.setAttribute('data-theme', this.current);
        }
    };

    // Intro screen
    const intro = {
        show() {
            const introScreen = document.getElementById('introScreen');
            if (introScreen) {
                introScreen.style.display = 'flex';
                introScreen.classList.remove('hidden');
            }
        },

        hide() {
            const introScreen = document.getElementById('introScreen');
            if (introScreen) {
                introScreen.classList.add('hidden');
                setTimeout(() => {
                    if (introScreen) {
                        introScreen.style.display = 'none';
                    }
                }, 600);
            }
        },

        init() {
            this.show();
            setTimeout(() => {
                this.hide();
            }, 3000);
        }
    };

    function init() {
        try {
            theme.init();
            intro.init();
            setTimeout(() => {
                initDOM();
                auth.checkAuth();
                storage.load();
                setupEvents();
                chatMgr.updateHistoryView();
                if (domRefs.messageInput) {
                    ui.adjustTextareaHeight(domRefs.messageInput);
                }
                updateAIStatus();
            }, 3100);
        } catch (error) {
            console.error('Init error:', error);
        }
    }

    // Start application when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
