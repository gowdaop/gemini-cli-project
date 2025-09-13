// Chat Page Controller - Performance Optimized
class ChatPage {
    constructor() {
        this.api = new ChatAPI();
        this.conversationId = null;
        this.isLoading = false;
        this.documentContext = null;
        
        // DOM elements
        this.chatMessages = null;
        this.chatInput = null;
        this.sendBtn = null;
        this.fileInput = null;
        this.sidebar = null;
        
        // Message history
        this.messages = [];
        
        // ✅ Performance: Debounce resize
        this.resizeDebounce = null;
        
        this.init();
    }

    async init() {
        try {
            this.initializeElements();
            this.setupEventListeners();
            this.loadInitialContext();
            this.renderRecentConversations(); // <-- ADDED
            
            // ✅ Performance: Simplified textarea resize
            this.setupTextareaResize();
            
            console.log('✅ Chat page initialized successfully');
        } catch (error) {
            console.error('❌ Chat page initialization failed:', error);
            this.showError('Failed to initialize chat. Please refresh the page.');
        }
    }

    initializeElements() {
        // Main elements
        this.chatMessages = DOM.id('chatMessages');
        this.chatInput = DOM.id('chatInput');
        this.sendBtn = DOM.id('sendBtn');
        this.fileInput = DOM.id('fileInput');
        this.sidebar = DOM.id('chatSidebar');
        this.conversationList = DOM.id('conversationList');
        
        // Control buttons
        this.attachFileBtn = DOM.id('attachFileBtn');
        this.toggleSidebarBtn = DOM.id('toggleSidebarBtn');
        this.clearChatBtn = DOM.id('clearChatBtn');
        this.newChatBtn = DOM.id('newChatBtn');
        this.uploadDocBtn = DOM.id('uploadDocBtn');
        this.viewResultsBtn = DOM.id('viewResultsBtn');
        
        // Context elements
        this.documentContextContainer = DOM.id('documentContext');
        
        if (!this.chatMessages || !this.chatInput || !this.sendBtn) {
            throw new Error('Required DOM elements not found');
        }
    }

    setupEventListeners() {
        // Send message
        this.sendBtn?.addEventListener('click', () => this.handleSendMessage());
        
        // Input handling - ✅ Optimized with debouncing
        this.chatInput?.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.chatInput?.addEventListener('input', this.debounce(() => this.updateSendButton(), 100));
        
        // File handling
        this.attachFileBtn?.addEventListener('click', () => this.fileInput?.click());
        this.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Controls
        this.toggleSidebarBtn?.addEventListener('click', () => this.toggleSidebar());
        this.clearChatBtn?.addEventListener('click', () => this.clearConversation());
        
        // Quick actions
        this.newChatBtn?.addEventListener('click', () => this.startNewConversation());
        this.uploadDocBtn?.addEventListener('click', () => window.location.href = 'upload.html');
        this.viewResultsBtn?.addEventListener('click', () => window.location.href = 'results.html');
        
        // Suggestion buttons
        this.setupSuggestionButtons();
    }

    // ✅ Performance: Debounce utility
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    setupSuggestionButtons() {
        // ✅ Use event delegation for better performance
        this.chatMessages?.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const question = e.target.dataset.question;
                if (question) {
                    this.chatInput.value = question;
                    this.handleSendMessage();
                }
            }
        });
    }

    setupTextareaResize() {
        if (!this.chatInput) return;
        
        // ✅ Simplified resize logic
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 100) + 'px';
        });
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSendMessage();
        }
    }

    updateSendButton() {
        const hasText = this.chatInput?.value.trim().length > 0;
        if (this.sendBtn) {
            this.sendBtn.disabled = !hasText || this.isLoading;
        }
    }

    async handleSendMessage() {
        const message = this.chatInput?.value.trim();
        if (!message || this.isLoading) return;

        try {
            this.isLoading = true;
            this.updateSendButton();
            
            // Clear welcome message
            this.clearWelcomeMessage();
            
            // Add user message
            this.addMessage('user', message);
            this.chatInput.value = '';
            this.chatInput.style.height = 'auto'; // Reset height
            this.updateSendButton();
            
            // Show typing indicator
            this.showTypingIndicator();
            
            // ✅ FIXED: Better error handling for API calls
            const response = await this.api.sendMessage(
                message, 
                this.conversationId,
                this.documentContext?.ocr,
                this.documentContext?.summary
            );
            
            // Update conversation ID
            this.conversationId = response.conversation_id;
            
            // Remove typing indicator and add response
            this.hideTypingIndicator();
            this.addMessage('bot', response.answer, response.evidence);

            this.saveConversation();
            
        } catch (error) {
            console.error('Send message error:', error);
            this.hideTypingIndicator();
            
            // ✅ Better error messages
            let errorMessage = 'Sorry, I encountered an error. Please try again.';
            if (error.message.includes('timed out')) {
                errorMessage = 'The request timed out. Please check your connection and try again.';
            } else if (error.message.includes('Failed to fetch')) {
                errorMessage = 'Unable to connect to the server. Please check your connection.';
            }
            
            this.addMessage('bot', errorMessage);
        } finally {
            this.isLoading = false;
            this.updateSendButton();
            this.scrollToBottom();
        }
    }

    // ✅ Optimized message rendering
    addMessage(sender, text, evidence = null) {
        if (!this.chatMessages) return;

        const messageEl = document.createElement('div');
        messageEl.classList.add('message', sender);
        
        const contentEl = document.createElement('div');
        contentEl.classList.add('message-content');
        
        // Simple text handling - no heavy processing
        if (sender === 'bot') {
            contentEl.innerHTML = this.formatBotMessage(text);
        } else {
            contentEl.textContent = text;
        }
        
        const timeEl = document.createElement('div');
        timeEl.classList.add('message-time');
        timeEl.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageEl.appendChild(contentEl);
        messageEl.appendChild(timeEl);
        
        this.chatMessages.appendChild(messageEl);
        this.messages.push({ sender, text, timestamp: Date.now() });
        
        // ✅ Optimized scrolling
        requestAnimationFrame(() => this.scrollToBottom());
    }

    // ✅ Simplified message formatting
    formatBotMessage(text) {
        // Basic formatting only - avoid heavy regex
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\n/g, '<br>');
        return text;
    }

    showTypingIndicator() {
        const typingEl = document.createElement('div');
        typingEl.classList.add('message', 'bot', 'typing-indicator');
        typingEl.innerHTML = '
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        ';
        typingEl.id = 'typingIndicator';
        this.chatMessages?.appendChild(typingEl);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        DOM.id('typingIndicator')?.remove();
    }

    clearWelcomeMessage() {
        const welcomeEl = DOM.qs('.welcome-message');
        if (welcomeEl) {
            welcomeEl.style.display = 'none';
        }
    }

    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }

    toggleSidebar() {
        if (window.innerWidth <= 768) {
            this.sidebar?.classList.toggle('visible');
        } else {
            this.sidebar?.classList.toggle('hidden');
        }
    }

    async clearConversation() {
        if (this.conversationId) {
            try {
                await this.api.deleteConversation(this.conversationId);
            } catch (error) {
                console.error('Delete conversation error:', error);
            }
        }
        
        this.startNewConversation();
    }

    startNewConversation() {
        this.conversationId = null;
        this.messages = [];
        
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <i class="fas fa-balance-scale"></i>
                    </div>
                    <h2>Welcome to Legal AI Assistant</h2>
                    <p>Ask me anything about your legal documents, contract terms, or risk assessments.</p>
                    <div class="suggested-questions">
                        <button class="suggestion-btn" data-question="What are the main risks in my contract?">
                            <i class="fas fa-exclamation-triangle"></i>
                            What are the main risks in my contract?
                        </button>
                        <button class="suggestion-btn" data-question="Explain this liability clause">
                            <i class="fas fa-shield-alt"></i>
                            Explain this liability clause
                        </button>
                        <button class="suggestion-btn" data-question="How can I reduce legal risks?">
                            <i class="fas fa-lightbulb"></i>
                            How can I reduce legal risks?
                        </button>
                    </div>
                </div>
            ';
        }
    }

    loadInitialContext() {
        const urlParams = new URLSearchParams(window.location.search);
        const filename = urlParams.get('filename');

        if (filename) {
            this.loadDocumentFromStorage(filename);
        } else {
            this.showAddContextUI();
        }
    }

    loadDocumentFromStorage(filename) {
        try {
            const analysisData = localStorage.getItem('latest_analysis');
            if (analysisData) {
                const data = JSON.parse(analysisData);
                if (data.filename === filename) {
                    this.documentContext = {
                        filename: data.filename,
                        ocr: data.ocr,
                        summary: data.analysis?.summary_200w
                    };
                    this.updateDocumentContextUI();
                } else {
                    this.showAddContextUI("Specified document not found. You can still ask general questions or add a different document.");
                }
            } else {
                this.showAddContextUI("No document context found. You can ask general legal questions.");
            }
        } catch (error) {
            console.error('Load document context error:', error);
            this.showAddContextUI("Error loading document context.");
        }
    }

    updateDocumentContextUI() {
        if (!this.documentContext || !this.documentContextContainer) return;
        
        this.documentContextContainer.innerHTML = '
            <div class="document-info">
                <div class="document-icon">
                    <i class="fas fa-file-contract"></i>
                </div>
                <div class="document-details">
                    <h4>' + this.documentContext.filename + '</h4>
                    <p>Document loaded and ready for context-aware responses</p>
                </div>
            </div>
        ';
    }

    showAddContextUI(message = "Ask general legal questions, or add a document for context-aware chat.") {
        if (!this.documentContextContainer) return;
        this.documentContextContainer.innerHTML = '
            <div class="add-context">
                <p>' + message + '</p>
                <button class="btn btn-secondary" id="addContextBtn">Add Document</button>
            </div>
        ';
        const addContextBtn = DOM.id('addContextBtn');
        if (addContextBtn) {
            addContextBtn.addEventListener('click', () => {
                window.location.href = 'upload.html';
            });
        }
    }

    saveConversation() {
        if (!this.conversationId || this.messages.length === 0) return;

        let history = [];
        try {
            const storedHistory = localStorage.getItem('chat_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }
        } catch (e) {
            console.error("Failed to parse chat history", e);
            history = [];
        }

        const conversation = {
            id: this.conversationId,
            messages: this.messages,
            timestamp: Date.now(),
            title: this.messages[0].text.substring(0, 30) + "..."
        };

        // Check if conversation already exists and update it
        const existingIndex = history.findIndex(c => c.id === this.conversationId);
        if (existingIndex > -1) {
            history[existingIndex] = conversation;
        } else {
            history.unshift(conversation);
        }

        // Keep only the last 5
        history = history.slice(0, 5);

        localStorage.setItem('chat_history', JSON.stringify(history));
        this.renderRecentConversations();
    }

    renderRecentConversations() {
        if (!this.conversationList) return;

        let history = [];
        try {
            const storedHistory = localStorage.getItem('chat_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }
        } catch (e) {
            console.error("Failed to parse chat history", e);
            history = [];
        }

        if (history.length === 0) {
            this.conversationList.innerHTML = '
                <div class="no-conversations">
                    <i class="fas fa-comment-slash"></i>
                    <p>No conversations yet</p>
                </div>
            ';
            return;
        }

        this.conversationList.innerHTML = ''; // Clear existing
        history.forEach(conv => {
            const itemEl = document.createElement('div');
            itemEl.className = 'conversation-item';
            itemEl.dataset.conversationId = conv.id;
            itemEl.innerHTML = '
                <div class="conversation-title">' + conv.title + '</div>
                <div class="conversation-timestamp">' + new Date(conv.timestamp).toLocaleString() + '</div>
            ';
            this.conversationList.appendChild(itemEl);
        });

        // Add event listeners for the new items
        this.conversationList.querySelectorAll('.conversation-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const conversationId = e.currentTarget.dataset.conversationId;
                this.loadConversation(conversationId);
            });
        });
    }

    loadConversation(conversationId) {
        let history = [];
        try {
            const storedHistory = localStorage.getItem('chat_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }
        } catch (e) {
            console.error("Failed to parse chat history", e);
            return;
        }

        const conversation = history.find(c => c.id === conversationId);
        if (conversation) {
            this.conversationId = conversation.id;
            this.messages = conversation.messages;
            
            this.chatMessages.innerHTML = ''; // Clear chat window
            this.messages.forEach(msg => {
                this.addMessage(msg.sender, msg.text);
            });
        }
    }

    async handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        this.addMessage('bot', `I've received your file "${file.name}". Please upload it through the main upload page for full analysis, then return here for document-specific questions.`);
    }

    showError(message) {
        console.error('Chat error:', message);
        // Simple error display
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = '
            position: fixed; top: 100px; left: 50%; transform: translateX(-50%);
            background: #fee; border: 1px solid #fcc; color: #c33;
            padding: 1rem; border-radius: 6px; z-index: 1000;
        ';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => errorDiv.remove(), 5000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.chatPage = new ChatPage();
        console.log('✅ Chat page initialized');
    } catch (error) {
        console.error('❌ Failed to initialize chat page:', error);
    }
});
