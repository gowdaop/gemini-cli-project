// Professional Legal Chat Interface Controller
class LegalChatPage {
  constructor() {
    // Core state
    this.api = new ChatAPI();
    this.conversationId = null;
    this.isLoading = false;
    this.documentContext = null;
    this.recentDocuments = [];

    // Context selection state
    this.selectedClauseId = null;
    this.selectedRiskLevel = null;
    this.analysisResults = null; // Store clauses and risks

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
      this.hydrateContextFromStorage();
      console.log('✅ Legal Chat page initialized successfully');
    } catch (error) {
      console.error('❌ Chat page initialization failed:', error);
      this.showError('Failed to initialize chat. Please refresh the page.');
    }
  }

  initializeElements() {
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
      recentDocsDropdown: DOM.id('recentDocsDropdown'),
      recentDocsContent: DOM.id('recentDocsContent'),
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
    this.elements.sendBtn?.addEventListener('click', () => this.handleSendMessage());
    this.elements.chatInput?.addEventListener('keydown', (e) => this.handleInputKeydown(e));
    this.elements.chatInput?.addEventListener('input', () => this.updateSendButton());
    this.elements.clearChatBtn?.addEventListener('click', () => this.clearChat());

    // Context management
    this.elements.removeContextBtn?.addEventListener('click', () => {
      this.removeDocumentContext();
      this.updateContextIndicator();
    });

    // Recent documents
    this.elements.recentDocsBtn?.addEventListener('click', () => this.toggleRecentDocsDropdown());

    // File upload passthrough
    this.elements.uploadBtn?.addEventListener('click', () => this.elements.fileInput?.click());
    this.elements.fileInput?.addEventListener('change', (e) => this.handleFileUpload(e));

    // Suggestions
    this.setupSuggestionButtons();

    // Close dropup on outside click
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

  getValidOCRPayload() {
    const ocr = this.documentContext?.ocr;
    // Only include OCR when blocks array is non-empty and first item has text field
    if (ocr && Array.isArray(ocr.blocks) && ocr.blocks.length > 0 && 
        ocr.blocks[0] && typeof ocr.blocks[0].text === 'string' && ocr.blocks[0].text.trim()) {
      return ocr;
    }
    return null;
  }

  async ensureConversation() {
    if (!this.conversationId) {
      const res = await this.api.newConversation();
      this.conversationId = res?.conversation_id || null;
      
      // If we have analysis results, store them immediately
      if (this.conversationId && this.analysisResults) {
        await this.storeAnalysisInConversation();
      }
    }
  }

  async handleSendMessage() {
    const message = this.elements.chatInput?.value.trim();
    if (!message || this.isLoading) return;

    try {
      this.isLoading = true;
      this.updateSendButton();

      this.clearWelcomeMessage();
      this.addMessage('user', message);

      this.elements.chatInput.value = '';
      this.elements.chatInput.style.height = 'auto';
      this.updateSendButton();

      this.showTypingIndicator();

      await this.ensureConversation();

      const ocrPayload = this.getValidOCRPayload();
      const normalizedRiskLevel = this.selectedRiskLevel ? this.selectedRiskLevel.toLowerCase() : null;
      
      const response = await this.api.sendMessage(
        message,
        this.conversationId,
        ocrPayload,
        this.documentContext?.summary || null,
        this.selectedClauseId,
        normalizedRiskLevel
      );

      this.conversationId = response.conversation_id;
      this.hideTypingIndicator();
      this.addMessage('bot', response.answer, response.evidence);
    } catch (error) {
      console.error('❌ Send message error:', error);
      this.hideTypingIndicator();

      let errorMessage = 'Sorry, I encountered an error. Please try again.';
      if (error.message?.includes('timed out')) {
        errorMessage = 'The request timed out. Please check your connection and try again.';
      } else if (error.message?.includes('Failed to fetch')) {
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
      // Minimal formatting: allow strong/em via ** and *
      const html = String(text || '')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
      contentEl.innerHTML = html;
    } else {
      contentEl.textContent = text;
    }

    const timeEl = document.createElement('div');
    timeEl.classList.add('message-time');
    timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageEl.appendChild(contentEl);
    messageEl.appendChild(timeEl);

    if (Array.isArray(evidence) && evidence.length > 0) {
      const evidenceEl = this.createEvidenceElement(evidence);
      messageEl.appendChild(evidenceEl);
    }

    this.elements.chatMessages.appendChild(messageEl);
    this.messages.push({ sender, text, timestamp: Date.now() });

    this.scrollToBottom();
  }

  createEvidenceElement(evidence) {
    const evidenceEl = document.createElement('div');
    evidenceEl.className = 'message-evidence';
    evidenceEl.innerHTML = `
      <div class="evidence-header">
        <i class="fas fa-book-open"></i>
        <span>Legal References (${evidence.length})</span>
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
          <span></span><span></span><span></span>
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
    if (welcomeEl) welcomeEl.style.display = 'none';
  }

  scrollToBottom() {
    if (this.elements.chatMessages) {
      setTimeout(() => {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
      }, 0);
    }
  }

  async clearChat() {
    try {
      if (this.conversationId) {
        await this.api.deleteConversation(this.conversationId);
      }
      this.conversationId = null;
      this.messages = [];
      this.selectedClauseId = null;
      this.selectedRiskLevel = null;
      this.analysisResults = null;
      this.hideContextSelectionUI();

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

      if (this.elements.contextIndicator) this.elements.contextIndicator.style.display = 'none';
      if (this.elements.recentDocsBtn) this.elements.recentDocsBtn.classList.remove('active');
      console.log('✅ Chat cleared successfully');
    } catch (error) {
      console.error('❌ Clear chat error:', error);
    }
  }

  // Document Context Management
  async setDocumentContext(documentData) {
    this.documentContext = documentData;

    if (this.elements.contextIndicator && this.elements.contextFilename) {
      this.elements.contextFilename.textContent = `Context: ${documentData.filename}`;
      this.elements.contextIndicator.style.display = 'block';
    }
    if (this.elements.recentDocsBtn) this.elements.recentDocsBtn.classList.add('active');

    // Create conversation before storing analysis
    await this.ensureConversation();

    if (documentData?.analysis?.clauses?.length || documentData?.analysis?.risks?.length) {
      await this.setAnalysisResults({
        clauses: documentData.analysis.clauses || [],
        risks: documentData.analysis.risks || []
      });
    } else if (documentData?.clauses?.length || documentData?.risks?.length) {
      await this.setAnalysisResults({
        clauses: documentData.clauses || [],
        risks: documentData.risks || []
      });
    }

    this.updateContextIndicator();
    console.log('✅ Document context set:', documentData.filename);
  }

  removeDocumentContext() {
    this.documentContext = null;
    if (this.elements.contextIndicator) this.elements.contextIndicator.style.display = 'none';
    if (this.elements.recentDocsBtn) this.elements.recentDocsBtn.classList.remove('active');
    console.log('✅ Document context removed');
  }

  // Recent Documents
  loadRecentDocuments() {
    try {
      const recentAnalyses = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.includes('analysis') || key === 'latest_analysis')) {
          try {
            const data = JSON.parse(localStorage.getItem(key));
            if (data && data.filename && data.timestamp) {
              recentAnalyses.push(data);
            }
          } catch (e) { /* skip */ }
        }
      }
      this.recentDocuments = recentAnalyses
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        .slice(0, 10);
      console.log(`✅ Loaded ${this.recentDocuments.length} recent documents`);
    } catch (error) {
      console.error('❌ Load recent documents error:', error);
      this.recentDocuments = [];
    }
  }

  toggleRecentDocsDropdown() {
    if (!this.elements.recentDocsContent) return;
    const isVisible = this.elements.recentDocsContent.style.display !== 'none';
    if (isVisible) this.hideRecentDocsDropdown();
    else this.showRecentDocsDropdown();
  }

  showRecentDocsDropdown() {
    if (!this.elements.recentDocsContent || !this.elements.recentDocsList) return;
    this.loadRecentDocuments();
    this.renderRecentDocuments();
    this.elements.recentDocsContent.style.display = 'block';
    console.log('✅ Recent docs dropdown shown');
  }

  hideRecentDocsDropdown() {
    if (this.elements.recentDocsContent) this.elements.recentDocsContent.style.display = 'none';
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

    this.elements.recentDocsList.querySelectorAll('.recent-doc-item').forEach(item => {
      item.addEventListener('click', () => {
        const filename = item.dataset.filename;
        const docData = this.recentDocuments.find(doc => doc.filename === filename);
        if (docData) {
          this.setDocumentContext(docData);
          this.hideRecentDocsDropdown();
        }
      });
    });
  }

  handleOutsideClick(e) {
    if (!this.elements.recentDocsDropdown) return;
    const isDropdownVisible = this.elements.recentDocsContent.style.display !== 'none';
    const isClickInsideDropdown = this.elements.recentDocsDropdown.contains(e.target);
    if (isDropdownVisible && !isClickInsideDropdown) {
      this.hideRecentDocsDropdown();
    }
  }

  async handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      this.showLoading('Uploading and analyzing document...');
      this.addMessage('bot', `I've received your file "${file.name}". Please use the main upload page for full document analysis, then return here for context-aware questions.`);
    } catch (error) {
      console.error('❌ File upload error:', error);
      this.addMessage('bot', 'Sorry, I encountered an error uploading your file. Please try again.');
    } finally {
      this.hideLoading();
      e.target.value = '';
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

  // Accept analysis results, persist to conversation, show selector UI
  async setAnalysisResults(analysisData) {
    this.analysisResults = analysisData;
    await this.ensureConversation();
    await this.storeAnalysisInConversation();
    this.showContextSelectionUI();
    this.updateContextIndicator();
  }

  async storeAnalysisInConversation() {
    if (this.conversationId && this.analysisResults && 
        (this.analysisResults?.clauses?.length || this.analysisResults?.risks?.length)) {
      try {
        await this.api.storeAnalysisResults(
          this.conversationId,
          this.analysisResults.clauses || [],
          this.analysisResults.risks || []
        );
        console.log('✅ Analysis results stored in conversation');
      } catch (error) {
        console.error('❌ Failed to store analysis results:', error);
      }
    }
  }

  // Inject context selection panel
  showContextSelectionUI() {
    if (!this.analysisResults || !this.elements.chatMessages) return;
    this.hideContextSelectionUI();

    const contextPanel = document.createElement('div');
    contextPanel.className = 'context-selection-panel';
    contextPanel.innerHTML = `
      <div class="context-panel-header">
        <h3><i class="fas fa-filter"></i> Select Context for Questions</h3>
        <button class="close-context-panel" id="closeContextPanel" aria-label="Close context panel" title="Close">
          <i class="fas fa-times"></i>
        </button>
      </div>

      <div class="context-tabs">
        <button class="context-tab active" data-tab="clauses">
          <i class="fas fa-file-contract"></i>
          Clauses (${this.analysisResults.clauses?.length || 0})
        </button>
        <button class="context-tab" data-tab="risks">
          <i class="fas fa-exclamation-triangle"></i>
          Risks by Level
        </button>
      </div>

      <div class="context-content">
        <div class="context-tab-content active" id="clausesTab">
          ${this.renderClausesList()}
        </div>
        <div class="context-tab-content" id="risksTab">
          ${this.renderRiskLevels()}
        </div>
      </div>
    `;

    this.elements.chatMessages.insertBefore(contextPanel, this.elements.chatMessages.firstChild);
    this.setupContextPanelListeners(contextPanel);
  }

  renderClausesList() {
    if (!this.analysisResults?.clauses || this.analysisResults.clauses.length === 0) {
      return '<p>No clauses found.</p>';
    }
    return `
      <div class="clauses-list">
        ${this.analysisResults.clauses.map(clause => `
          <div class="clause-item ${this.selectedClauseId === clause.id ? 'selected' : ''}"
               data-clause-id="${clause.id}">
            <div class="clause-tag">${String(clause.tag || '').replace('_', ' ').toUpperCase()}</div>
            <div class="clause-text">${this.truncateText(String(clause.text || ''), 100)}</div>
            <div class="clause-location">Page ${clause?.span?.page || 1}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  renderRiskLevels() {
    if (!this.analysisResults?.risks || this.analysisResults.risks.length === 0) {
      return '<p>No risks found.</p>';
    }
    const risksByLevel = this.groupRisksByLevel(this.analysisResults.risks);
    return `
      <div class="risk-levels">
        ${Object.entries(risksByLevel).map(([level, risks]) => `
          <div class="risk-level-item ${this.selectedRiskLevel === level ? 'selected' : ''}"
               data-risk-level="${level}">
            <div class="risk-level-header">
              <span class="risk-indicator ${level}"></span>
              <span class="risk-level-name">${String(level || '').toUpperCase()} RISK</span>
              <span class="risk-count">(${risks.length} items)</span>
            </div>
            <div class="risk-summary">
              ${risks.length} clause${risks.length !== 1 ? 's' : ''} with ${String(level)} risk level
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  groupRisksByLevel(risks) {
    const grouped = {};
    risks.forEach(risk => {
      const level = risk?.level || 'unknown';
      if (!grouped[level]) grouped[level] = [];
      grouped[level].push(risk);
    });
    return grouped;
  }

  setupContextPanelListeners(panel) {
    panel.querySelectorAll('.context-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        panel.querySelectorAll('.context-tab').forEach(t => t.classList.remove('active'));
        panel.querySelectorAll('.context-tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        const tabName = tab.dataset.tab;
        panel.querySelector(`#${tabName}Tab`).classList.add('active');
      });
    });

    panel.querySelectorAll('.clause-item').forEach(item => {
      item.addEventListener('click', () => {
        const clauseId = item.dataset.clauseId;
        this.selectClause(clauseId, panel);
      });
    });

    panel.querySelectorAll('.risk-level-item').forEach(item => {
      item.addEventListener('click', () => {
        const riskLevel = item.dataset.riskLevel;
        this.selectRiskLevel(riskLevel, panel);
      });
    });

    panel.querySelector('#closeContextPanel')?.addEventListener('click', () => {
      this.hideContextSelectionUI();
    });
  }

  selectClause(clauseId, panel) {
    // Clear risk selection
    this.selectedRiskLevel = null;
    panel.querySelectorAll('.risk-level-item').forEach(item => item.classList.remove('selected'));

    // Toggle clause selection
    if (this.selectedClauseId === clauseId) {
      this.selectedClauseId = null;
      panel.querySelectorAll('.clause-item').forEach(item => item.classList.remove('selected'));
    } else {
      this.selectedClauseId = clauseId;
      panel.querySelectorAll('.clause-item').forEach(item => item.classList.remove('selected'));
      panel.querySelector(`[data-clause-id="${clauseId}"]`)?.classList.add('selected');
    }
    this.updateContextIndicator();
  }

  selectRiskLevel(riskLevel, panel) {
    // Clear clause selection
    this.selectedClauseId = null;
    panel.querySelectorAll('.clause-item').forEach(item => item.classList.remove('selected'));

    // Normalize risk level to lowercase
    const normalizedRiskLevel = riskLevel ? riskLevel.toLowerCase() : null;

    // Toggle risk level selection
    if (this.selectedRiskLevel === normalizedRiskLevel) {
      this.selectedRiskLevel = null;
      panel.querySelectorAll('.risk-level-item').forEach(item => item.classList.remove('selected'));
    } else {
      this.selectedRiskLevel = normalizedRiskLevel;
      panel.querySelectorAll('.risk-level-item').forEach(item => item.classList.remove('selected'));
      panel.querySelector(`[data-risk-level="${riskLevel}"]`)?.classList.add('selected');
    }
    this.updateContextIndicator();
  }

  updateContextIndicator() {
    if (!this.elements.contextIndicator || !this.elements.contextFilename) return;

    if (this.selectedClauseId && this.analysisResults?.clauses) {
      const clause = this.analysisResults.clauses.find(c => String(c.id) === String(this.selectedClauseId));
      if (clause) {
        this.elements.contextFilename.textContent = `Selected: ${String(clause.tag || '').replace('_', ' ')} clause`;
        this.elements.contextIndicator.style.display = 'block';
        return;
      }
    }
    if (this.selectedRiskLevel) {
      this.elements.contextFilename.textContent = `Selected: ${String(this.selectedRiskLevel).toUpperCase()} risk items`;
      this.elements.contextIndicator.style.display = 'block';
      return;
    }
    if (this.documentContext?.filename) {
      this.elements.contextFilename.textContent = `Context: ${this.documentContext.filename}`;
      this.elements.contextIndicator.style.display = 'block';
      return;
    }
    this.elements.contextIndicator.style.display = 'none';
  }

  hideContextSelectionUI() {
    document.querySelector('.context-selection-panel')?.remove();
  }

  hydrateContextFromStorage() {
    try {
      const stored = localStorage.getItem('latest_analysis');
      if (stored) {
        const data = JSON.parse(stored);
        if (data?.filename) this.setDocumentContext(data);
        if (data?.analysis?.clauses || data?.analysis?.risks) {
          this.setAnalysisResults({
            clauses: data.analysis.clauses || [],
            risks: data.analysis.risks || []
          });
        }
      }
    } catch (e) {
      console.warn('Context hydration skipped:', e);
    }
  }

  // Utils
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
