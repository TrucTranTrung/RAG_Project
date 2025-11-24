// Chat app main controller
(function () {
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
                USE_OLLAMA: window.CHAT_CONFIG.ollama?.enabled || true,
                UPLOAD_API_URL: window.CHAT_CONFIG.upload?.apiUrl || 'http://localhost:8001/upload'
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
        save: function () {
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

        load: function () {
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

        saveUsers: function (users) {
            try {
                localStorage.setItem(STORAGE_KEYS.USERS, JSON.stringify(users));
            } catch (error) {
                console.warn('Failed to save users:', error);
            }
        },

        loadUsers: function () {
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
        createMessageElement: function (role, content, attachments) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.setAttribute('role', 'listitem');

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === MESSAGE_ROLES.USER ? 'B' : 'AI';
            avatar.setAttribute('aria-label', role === MESSAGE_ROLES.USER ? 'Người dùng' : 'Trợ lý AI');

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            // 处理附件 (video/audio)
            if (attachments) {
                const attachmentsDiv = document.createElement('div');
                attachmentsDiv.className = 'message-attachments';

                // 处理视频
                if (attachments.videos && attachments.videos.length > 0) {
                    attachments.videos.forEach(videoSrc => {
                        const videoWrapper = document.createElement('div');
                        videoWrapper.className = 'message-video-wrapper';
                        const videoEl = document.createElement('video');
                        videoEl.src = videoSrc;
                        videoEl.className = 'message-video';
                        videoEl.controls = true;
                        videoEl.preload = 'metadata';
                        videoWrapper.appendChild(videoEl);
                        attachmentsDiv.appendChild(videoWrapper);
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
        open: function () {
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
        generateResponse: async function (userMessage, audio, videos, conversationHistory, onStream) {
            console.log('[AI Service] Đang sử dụng Gemini AI');
            return await this.generateWithOllama(userMessage, audio, videos, conversationHistory, onStream);
        },

        async generateWithOllama(userMessage, audio, videos, conversationHistory, onStream) {
            try {
                console.log('[Bot] Đang gửi request đến bot...');
                const startTime = Date.now();

                // Gửi trực tiếp userMessage (không gộp history) vì bot chấp nhận string
                const formData = new FormData();
                formData.append('question', userMessage);
                if (audio) {
                    const audioFile = await this.getFileFromUrl(audio, `audio_${Date.now()}.mp3`, 'audio/mpeg');
                    formData.append('audio', audioFile);
                }
                if (videos && videos.length > 0) {
                    for (let i = 0; i < videos.length; i++) {
                        const videoFile = await this.getFileFromUrl(videos[i], `video_${Date.now()}_${i}.mp4`, 'video/mp4');
                        formData.append('audio', videoFile);
                    }
                }

                const response = await fetch(`${CONFIG.OLLAMA_API_URL}`, {
                    method: 'POST',
                    body: formData
                });

                // const response = {
                //     request_id: 'dummy_id_12345',
                //     audio_base64: "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//PkZAAAAAGkAAAAAAAAA0gAAAAATEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
                // }

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Bot API error: ${response.status} - ${errorText}`);
                }

                // Backend trả JSON:
                const data = await response.json();
                // const data = response;  

                // Nếu backend trả audio_base64, trả về base64 (caller xử lý)
                if (data.audio_base64) {
                    const elapsedTime = Date.now() - startTime;
                    console.log(`[Bot] Received audio (base64) — elapsed ${elapsedTime}ms`);
                    return {
                        type: 'audio',
                        audio_base64: data.audio_base64,
                        request_id: data.request_id || null
                    };
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

        async getFileFromUrl(url, filename, mimeType) {
            const response = await fetch(url);
            const blob = await response.blob();
            return new File([blob], filename, { type: mimeType });
        }

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
        createNew: function () {
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
        videos: [],
        audio: null, // { blob, preview, waveform, duration }
        recording: null,
        mediaRecorder: null,
        recordingStream: null,
        recordingTimer: null,
        recordingStartTime: null,

        hasAttachment: function () {
            return this.videos.length > 0 || this.audio !== null;
        },

        addVideo: function (file) {
            if (this.audio) {
                alert('Bạn đã chọn âm thanh. Chỉ được gửi 1 tệp (video hoặc âm thanh).');
                return;
            }
            // Chỉ giữ 1 video duy nhất
            if (this.videos.length > 0) {
                this.videos.forEach((vid) => {
                    if (vid.preview) URL.revokeObjectURL(vid.preview);
                });
                this.videos = [];
            }
            const preview = URL.createObjectURL(file);
            this.videos.push({ file, preview });
            this.updatePreview();
        },

        addAudioFile: async function (file) {
            if (this.videos.length > 0) {
                alert('Bạn đã chọn video. Chỉ được gửi 1 tệp (video hoặc âm thanh).');
                return;
            }
            if (this.audio?.preview) {
                URL.revokeObjectURL(this.audio.preview);
            }
            const preview = URL.createObjectURL(file);
            let waveform = null;
            let duration = null;
            try {
                const result = await this.generateWaveform(file);
                waveform = result.waveform;
                duration = result.duration;
            } catch (err) {
                console.warn('Cannot generate waveform for audio file:', err);
            }
            this.audio = {
                blob: file,
                preview,
                waveform,
                duration
            };
            this.updatePreview();
        },

        removeVideo: function (index) {
            const removed = this.videos.splice(index, 1);
            if (removed[0]?.preview) {
                URL.revokeObjectURL(removed[0].preview);
            }
            this.updatePreview();
        },

        startRecording: function () {
            if (this.hasAttachment()) {
                alert('Chỉ được chọn 1 tệp. Hãy bỏ video/âm thanh đã chọn trước khi ghi âm.');
                return;
            }
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

        stopRecording: function () {
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
                this.stopTimer();
                domRefs.recordingStatus.style.display = 'none';
                domRefs.recordBtn.classList.remove('recording');
            }
        },

        cancelRecording: function () {
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

        startTimer: function () {
            this.recordingTimer = setInterval(() => {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                domRefs.recordingTime.textContent =
                    `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            }, 1000);
        },

        stopTimer: function () {
            if (this.recordingTimer) {
                clearInterval(this.recordingTimer);
                this.recordingTimer = null;
            }
        },

        updatePreview: function () {
            domRefs.previewItems.innerHTML = '';

            if (this.videos.length === 0 && !this.audio) {
                domRefs.inputPreview.style.display = 'none';
                return;
            }

            domRefs.inputPreview.style.display = 'block';

            // 显示视频预览
            this.videos.forEach((videoObj, index) => {
                const previewItem = document.createElement('div');
                previewItem.className = 'preview-item preview-video';
                const video = document.createElement('video');
                video.src = videoObj.preview;
                video.controls = true;
                video.preload = 'metadata';
                video.className = 'preview-video-el';
                const removeBtn = document.createElement('button');
                removeBtn.className = 'preview-remove';
                removeBtn.innerHTML = '×';
                removeBtn.onclick = () => this.removeVideo(index);
                previewItem.appendChild(video);
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
                this.videos.length > 0 ||
                this.audio !== null;
            domRefs.sendButton.disabled = !hasContent;
        },

        clear: function () {
            this.videos.forEach((vid) => {
                if (vid.preview) {
                    URL.revokeObjectURL(vid.preview);
                }
            });
            this.videos = [];
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

        createAudioCard: function ({ src, waveform, duration }) {
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
        async uploadVideo(file) {
            const formData = new FormData();
            formData.append('type', 'video');
            formData.append('file', file, file.name || `video_${Date.now()}.mp4`);

            const response = await fetch(window.CHAT_CONFIG.upload?.apiUrl || 'http://localhost:8001/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Không thể tải video lên');
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.message || 'Tải video thất bại');
            }
            return data.payload;
        },

        async uploadAudio(blob) {
            const formData = new FormData();
            formData.append('type', 'audio');
            const audioFile = new File([blob], `audio_${Date.now()}.webm`, { type: 'audio/webm' });
            formData.append('file', audioFile);
            let response;
            try {
                response = await fetch(window.CHAT_CONFIG.upload?.apiUrl || 'http://localhost:8001/upload', {
                    method: 'POST',
                    body: formData
                });
            } catch (error) {
                console.error('Upload audio error:', error);
                throw new Error('Lỗi khi tải âm thanh lên');
            }

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
                videos: [],
                audio: null
            };

            if (state.videos.length > 0) {
                const uploads = state.videos.map((vid) => this.uploadVideo(vid.file));
                results.videos = await Promise.all(uploads);
            }

            if (state.audio?.blob) {
                results.audio = await this.uploadAudio(state.audio.blob);
            }

            return results;
        }
    };

    // Message handler
    const msgHandler = {
        send: async function () {
            const messageText = ui.sanitizeText(domRefs.messageInput.value);
            const hasAttachments = attachmentMgr.videos.length > 0 || attachmentMgr.audio !== null;

            if (!messageText && !hasAttachments) return;

            if (!appState.activeConversationId) {
                chatMgr.initializeNew(messageText || 'Tin nhắn có đính kèm');
            }

            domRefs.sendButton.disabled = true;
            domRefs.sendButton.classList.add('is-uploading');

            let uploadedAttachments = { videos: [], audio: null };
            if (hasAttachments) {
                try {
                    uploadedAttachments = await uploader.uploadAll({
                        videos: attachmentMgr.videos,
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
            if (uploadedAttachments.videos.length > 0) {
                attachments.videos = uploadedAttachments.videos.map(item => item.url);
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

                const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                msgRenderer.renderMessageWithTyping(MESSAGE_ROLES.ASSISTANT, messageId);
                // Lấy response từ Ollama
                const response = await ai.generateResponse(messageText, attachments.audio, attachments.videos, conversationHistory);

                // Nếu response là audio (object có audio_base64) hoặc string base64
                let audioBase64 = null;
                let requestId = null;
                if (response && typeof response === 'object' && response.audio_base64) {
                    audioBase64 = response.audio_base64;
                    requestId = response.request_id || null;
                } else if (response && this.isBase64(response)) {
                    audioBase64 = response.trim();
                }

                if (audioBase64) {
                    let playbackSrc = `data:audio/mpeg;base64,${audioBase64}`;
                    let audioDuration = null;
                    let audioWaveform = null;
                    let audioBlob = null;
                    try {
                        audioBlob = this.base64ToBlob(audioBase64, 'audio/mpeg');
                        const wf = await attachmentMgr.generateWaveform(audioBlob);
                        audioWaveform = wf.waveform || null;
                        audioDuration = wf.duration || null;
                    } catch (err) {
                        console.warn('Không tạo được waveform từ audio base64:', err);
                    }
                    try {
                        if (!audioBlob) {
                            audioBlob = this.base64ToBlob(audioBase64, 'audio/mpeg');
                        }
                        const uploaded = await uploader.uploadAudio(audioBlob);
                        if (uploaded?.url) {
                            playbackSrc = uploaded.url;
                            audioDuration = uploaded.duration || audioDuration || null;
                        }
                    } catch (err) {
                        console.warn('Không upload được audio base64, dùng data URI:', err);
                    }

                    const messageElement = document.getElementById(messageId);
                    if (messageElement) {
                        const contentDiv = messageElement.querySelector('.message-content');
                        if (contentDiv) {
                            contentDiv.innerHTML = '';
                            const card = attachmentMgr.createAudioCard({
                                src: playbackSrc,
                                waveform: audioWaveform,
                                duration: audioDuration
                            });
                            contentDiv.appendChild(card.container);
                        }
                    }
                    // Phát âm thanh ngay khi nhận được
                    const audioEl = new Audio(playbackSrc);
                    audioEl.play().catch(err => {
                        console.warn('Không thể tự động phát âm thanh:', err);
                    });
                    appState.messages[appState.activeConversationId].push({
                        role: MESSAGE_ROLES.ASSISTANT,
                        content: 'response audio',
                        attachments: {
                            audio: playbackSrc,
                            audioDuration: audioDuration || undefined,
                            audioWaveform: audioWaveform || undefined
                        },
                        request_id: requestId || undefined
                    });
                    storage.save();
                    return;
                }

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
        },

        isBase64(str) {
            if (!str || typeof str !== 'string') return false;

            try {
                const sanitized = str.replace(/\s+/g, '');
                return btoa(atob(sanitized)) === sanitized;
            } catch (err) {
                return false;
            }
        },

        base64ToBlob(base64, mimeType = '') {
            const sanitized = (base64 || '').replace(/\s+/g, '');
            const byteChars = atob(sanitized);
            const byteNumbers = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {
                byteNumbers[i] = byteChars.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType || 'application/octet-stream' });
        }
    };

    // Auth
    const auth = {
        showModal: function () {
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

        // Upload video/audio
        if (domRefs.imageBtn && domRefs.imageInput) {
            domRefs.imageBtn.addEventListener('click', () => {
                domRefs.imageInput.click();
            });

            domRefs.imageInput.addEventListener('change', (e) => {
                if (attachmentMgr.hasAttachment()) {
                    alert('Chỉ được gửi 1 tệp. Hãy xóa tệp hiện tại trước khi chọn tệp khác.');
                    e.target.value = '';
                    return;
                }
                const file = e.target.files && e.target.files[0];
                if (!file) return;

                const task = (() => {
                    if (file.type.startsWith('video/')) {
                        attachmentMgr.addVideo(file);
                        return Promise.resolve();
                    }
                    if (file.type.startsWith('audio/')) {
                        return attachmentMgr.addAudioFile(file);
                    }
                    return Promise.resolve();
                })();

                Promise.resolve(task).finally(() => {
                    e.target.value = ''; // reset để chọn lại cùng file
                });
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

        show: function () {
            if (domRefs.confirmModal) {
                domRefs.confirmModal.classList.add('show');
                document.body.style.overflow = 'hidden';
            }
        },

        hide: function () {
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
