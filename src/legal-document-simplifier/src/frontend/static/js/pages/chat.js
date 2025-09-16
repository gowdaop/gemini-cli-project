// Professional Legal Chat Interface Controller

class LegalChatPage {
    constructor() {
        this.api = new ChatAPI();
        this.conversationId = null;
        this.isLoading = false;
        this.documentContext = null;
        this.recentDocuments = [];
        
        // DOM elements
        this.elements = {};
        
        // Message history
        this.messages = [];
        
        this.init();
    }

    async init() {
        try {
            this.initializeElements();
            this.setupEventListeners();
            this.loadRecentDocuments();
            this.setupTextareaAutoResize();
            
            console.log('✅ Legal Chat page initialized successfully');
        } catch (error) {
            console.error('❌ Chat page initialization failed:', error);
            this.showError('Failed to initialize chat. Please refresh the page.');
        }
    }

    initializeElements() {
        // Main elements
        this.elements = {
            chatMessages: DOM.id('chatMessages'),
            chatInput: DOM.id('chatInput'),
            sendBtn: DOM.id('sendBtn'),
            clearChatBtn: DOM.id('clearChatBtn'),
            
            // Context elements
            contextIndicator: DOM.id('contextIndicator'),
            contextFilename: DOM.id('contextFilename'),
            removeContextBtn: DOM.id('removeContextBtn'),
            
            // Recent docs elements
            recentDocsBtn: DOM.id('recentDocsBtn'),
            recentUploadsDropup: DOM.id('recentUploadsDropup'),
            closeDropupBtn: DOM.id('closeDropupBtn'),
            recentDocsList: DOM.id('recentDocsList'),
            
            // Upload elements
            uploadBtn: DOM.id('uploadBtn'),
            fileInput: DOM.id('fileInput'),
            
            // Loading
            loadingOverlay: DOM.id('loadingOverlay')
        };

        if (!this.elements.chatMessages || !this.elements.chatInput || !this.elements.sendBtn) {
            throw new Error('Required DOM elements not found');
        }
    }

    setupEventListeners() {
        // Send message
        this.elements.sendBtn?.addEventListener('click', () => this.handleSendMessage());
        
        // Input handling
        this.elements.chatInput?.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.elements.chatInput?.addEventListener('input', () => this.updateSendButton());
        
        // Clear chat
        this.elements.clearChatBtn?.addEventListener('click', () => this.clearChat());
        
        // Context management
        this.elements.removeContextBtn?.addEventListener('click', () => this.removeDocumentContext());
        
        // Recent documents
        this.elements.recentDocsBtn?.addEventListener('click', () => this.toggleRecentDocsDropup());
        this.elements.closeDropupBtn?.addEventListener('click', () => this.hideRecentDocsDropup());
        
        // File upload
        this.elements.uploadBtn?.addEventListener('click', () => this.elements.fileInput?.click());
        this.elements.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // Suggestion buttons
        this.setupSuggestionButtons();
        
        // Click outside to close dropup
        document.addEventListener('click', (e) => this.handleOutsideClick(e));
    }

    setupSuggestionButtons() {
        this.elements.chatMessages?.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const question = e.target.dataset.question;
                if (question) {
                    this.elements.chatInput.value = question;
                    this.handleSendMessage();
                }
            }
        });
    }

    setupTextareaAutoResize() {
        if (!this.elements.chatInput) return;
        
        this.elements.chatInput.addEventListener('input', () => {
            this.elements.chatInput.style.height = 'auto';
            this.elements.chatInput.style.height = Math.min(this.elements.chatInput.scrollHeight, 120) + 'px';
        });
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.handleSendMessage();
        }
    }

    updateSendButton() {
        const hasText = this.elements.chatInput?.value.trim().length > 0;
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = !hasText || this.isLoading;
        }
    }

    async handleSendMessage() {
        const message = this.elements.chatInput?.value.trim();
        if (!message || this.isLoading) return;

        try {
            this.isLoading = true;
            this.updateSendButton();
            
            // Clear welcome message
            this.clearWelcomeMessage();
            
            // Add user message
            this.addMessage('user', message);
            
            // Clear input and reset height
            this.elements.chatInput.value = '';
            this.elements.chatInput.style.height = 'auto';
            this.updateSendButton();
            
            // Show typing indicator
            this.showTypingIndicator();
            
            // Send to API
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
            
        } catch (error) {
            console.error('❌ Send message error:', error);
            this.hideTypingIndicator();
            
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

    addMessage(sender, text, evidence = null) {
        if (!this.elements.chatMessages) return;

        const messageEl = document.createElement('div');
        messageEl.classList.add('message', sender);
        
        const contentEl = document.createElement('div');
        contentEl.classList.add('message-content');
        
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
        
        // Add evidence if available
        if (evidence && evidence.length > 0) {
            const evidenceEl = this.createEvidenceElement(evidence);
            messageEl.appendChild(evidenceEl);
        }
        
        this.elements.chatMessages.appendChild(messageEl);
        this.messages.push({ sender, text, timestamp: Date.now() });
        
        this.scrollToBottom();
    }

    formatBotMessage(text) {
        // Basic formatting for bot messages
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        text = text.replace(/\n/g, '<br>');
        return text;
    }

    createEvidenceElement(evidence) {
        const evidenceEl = document.createElement('div');
        evidenceEl.className = 'message-evidence';
        evidenceEl.innerHTML = `
            <div class="evidence-header">
                <i class="fas fa-book-open"></i>
                <span>Legal References (${evidence.length})</span>
            </div>
            <div class="evidence-items">
                ${evidence.slice(0, 3).map(item => `
                    <div class="evidence-item">
                        <div class="evidence-source">${item.doc_type || 'Legal Document'}</div>
                        <div class="evidence-snippet">${this.truncateText(item.content, 150)}</div>
                        <div class="evidence-relevance">${Math.round(item.similarity * 100)}% relevant</div>
                    </div>
                `).join('')}
            </div>
        `;
        return evidenceEl;
    }

    showTypingIndicator() {
        const typingEl = document.createElement('div');
        typingEl.classList.add('message', 'bot', 'typing-indicator');
        typingEl.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span style="margin-left: 0.5rem; color: #64748b;">AI is thinking...</span>
            </div>
        `;
        typingEl.id = 'typingIndicator';
        this.elements.chatMessages?.appendChild(typingEl);
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
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    async clearChat() {
        try {
            // Clear conversation on server if exists
            if (this.conversationId) {
                await this.api.deleteConversation(this.conversationId);
            }
            
            // Reset local state
            this.conversationId = null;
            this.messages = [];
            
            // Clear UI
            if (this.elements.chatMessages) {
                this.elements.chatMessages.innerHTML = `
                    <div class="welcome-message">
                        <div class="welcome-avatar">
                            <i class="fas fa-scale-balanced"></i>
                        </div>
                        <h2>Legal AI Assistant</h2>
                        <p>Ask me anything about legal matters, contract analysis, or risk assessment. Upload a document for context-aware responses.</p>
                        
                        <div class="suggested-questions">
                            <button class="suggestion-btn" data-question="What are common contract risks I should be aware of?">
                                <i class="fas fa-exclamation-triangle"></i>
                                What are common contract risks I should be aware of?
                            </button>
                            <button class="suggestion-btn" data-question="Explain liability clauses in simple terms">
                                <i class="fas fa-shield-alt"></i>
                                Explain liability clauses in simple terms
                            </button>
                            <button class="suggestion-btn" data-question="How can I protect my intellectual property?">
                                <i class="fas fa-lightbulb"></i>
                                How can I protect my intellectual property?
                            </button>
                        </div>
                    </div>
                `;
            }
            
            console.log('✅ Chat cleared successfully');
            
        } catch (error) {
            console.error('❌ Clear chat error:', error);
        }
    }

    // Document Context Management
    setDocumentContext(documentData) {
        this.documentContext = documentData;
        
        if (this.elements.contextIndicator && this.elements.contextFilename) {
            this.elements.contextFilename.textContent = `Context: ${documentData.filename}`;
            this.elements.contextIndicator.style.display = 'block';
        }
        
        // Update recent docs button to show active state
        if (this.elements.recentDocsBtn) {
            this.elements.recentDocsBtn.classList.add('active');
        }
        
        console.log('✅ Document context set:', documentData.filename);
    }

    removeDocumentContext() {
        this.documentContext = null;
        
        if (this.elements.contextIndicator) {
            this.elements.contextIndicator.style.display = 'none';
        }
        
        if (this.elements.recentDocsBtn) {
            this.elements.recentDocsBtn.classList.remove('active');
        }
        
        console.log('✅ Document context removed');
    }

    // Recent Documents Management
    loadRecentDocuments() {
        try {
            const recentAnalyses = [];
            
            // Load from localStorage
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.includes('analysis')) {
                    try {
                        const data = JSON.parse(localStorage.getItem(key));
                        if (data && data.filename && data.timestamp) {
                            recentAnalyses.push(data);
                        }
                    } catch (e) {
                        // Skip invalid entries
                    }
                }
            }
            
            // Sort by timestamp (newest first)
            this.recentDocuments = recentAnalyses
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .slice(0, 10); // Keep only last 10
            
            console.log(`✅ Loaded ${this.recentDocuments.length} recent documents`);
            
        } catch (error) {
            console.error('❌ Load recent documents error:', error);
            this.recentDocuments = [];
        }
    }

    toggleRecentDocsDropup() {
        if (!this.elements.recentUploadsDropup) return;
        
        const isVisible = this.elements.recentUploadsDropup.style.display !== 'none';
        
        if (isVisible) {
            this.hideRecentDocsDropup();
        } else {
            this.showRecentDocsDropup();
        }
    }

    showRecentDocsDropup() {
        if (!this.elements.recentUploadsDropup || !this.elements.recentDocsList) return;
        
        // Refresh recent documents
        this.loadRecentDocuments();
        
        // Populate the list
        this.renderRecentDocuments();
        
        // Show dropup
        this.elements.recentUploadsDropup.style.display = 'block';
        
        console.log('✅ Recent docs dropup shown');
    }

    hideRecentDocsDropup() {
        if (this.elements.recentUploadsDropup) {
            this.elements.recentUploadsDropup.style.display = 'none';
        }
    }

    renderRecentDocuments() {
        if (!this.elements.recentDocsList) return;
        
        if (this.recentDocuments.length === 0) {
            this.elements.recentDocsList.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: #64748b;">
                    <i class="fas fa-inbox" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                    <p>No recent documents found.</p>
                    <p style="font-size: 0.875rem;">Upload a document to get started!</p>
                </div>
            `;
            return;
        }
        
        this.elements.recentDocsList.innerHTML = this.recentDocuments
            .map(doc => `
                <div class="recent-doc-item" data-filename="${doc.filename}">
                    <div class="doc-icon">
                        <i class="fas fa-file-contract"></i>
                    </div>
                    <div class="doc-details">
                        <div class="doc-filename">${this.truncateText(doc.filename, 40)}</div>
                        <div class="doc-timestamp">${this.formatTimestamp(doc.timestamp)}</div>
                    </div>
                </div>
            `).join('');
        
        // Add click handlers
        this.elements.recentDocsList.querySelectorAll('.recent-doc-item').forEach(item => {
            item.addEventListener('click', () => {
                const filename = item.dataset.filename;
                const docData = this.recentDocuments.find(doc => doc.filename === filename);
                if (docData) {
                    this.setDocumentContext(docData);
                    this.hideRecentDocsDropup();
                }
            });
        });
    }

    handleOutsideClick(e) {
        if (!this.elements.recentUploadsDropup) return;
        
        const isDropupVisible = this.elements.recentUploadsDropup.style.display !== 'none';
        const isClickInsideDropup = this.elements.recentUploadsDropup.contains(e.target);
        const isClickOnButton = this.elements.recentDocsBtn?.contains(e.target);
        
        if (isDropupVisible && !isClickInsideDropup && !isClickOnButton) {
            this.hideRecentDocsDropup();
        }
    }

    async handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        try {
            this.showLoading('Uploading and analyzing document...');
            
            // Here you would integrate with your upload API
            // For now, just show a message
            this.addMessage('bot', `I've received your file "${file.name}". Please use the main upload page for full document analysis, then return here for context-aware questions.`);
            
        } catch (error) {
            console.error('❌ File upload error:', error);
            this.addMessage('bot', 'Sorry, I encountered an error uploading your file. Please try again.');
        } finally {
            this.hideLoading();
            e.target.value = ''; // Clear file input
        }
    }

    showLoading(message = 'Processing...') {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.querySelector('p').textContent = message;
            this.elements.loadingOverlay.style.display = 'flex';
        }
    }

    hideLoading() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
    }

    // Utility Methods
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        
        return date.toLocaleDateString();
    }

    showError(message) {
        console.error('Chat error:', message);
        // You could implement a toast notification here
        this.addMessage('bot', `System Error: ${message}`);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.legalChatPage = new LegalChatPage();
        console.log('✅ Legal Chat page initialized');
    } catch (error) {
        console.error('❌ Failed to initialize legal chat page:', error);
    }
});
