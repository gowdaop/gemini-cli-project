// Fixed Upload Process - frontend/static/js/pages/upload.js

class UploadPage {
    constructor() {
        this.dropzone = null;
        this.fileInput = null;
        this.currentFile = null;
        
        // ✅ FIX: Add dependency checks before instantiation
        this.validateDependencies();
        
        this.uploadAPI = new UploadAPI();
        this.analysisAPI = new AnalysisAPI();
        this.init();
    }

    // ✅ NEW: Validate all required dependencies
    validateDependencies() {
        const missing = [];
        
        if (typeof DOM === 'undefined') {
            missing.push('DOM utilities (helpers.js)');
        }
        
        if (typeof UploadAPI === 'undefined') {
            missing.push('UploadAPI class (api/upload.js)');
        }
        
        if (typeof AnalysisAPI === 'undefined') {
            missing.push('AnalysisAPI class (api/upload.js)');
        }
        
        if (missing.length > 0) {
            throw new Error(`Missing dependencies: ${missing.join(', ')}. Check script loading order.`);
        }
    }

    async init() {
        try {
            this.initializeElements();
            this.setupEventListeners();
            this.renderRecentUploads(); // <-- ADDED
            console.log('✅ UploadPage initialized successfully');
        } catch (error) {
            console.error('❌ UploadPage initialization failed:', error);
            this.showError('Failed to initialize upload page. Please refresh and try again.');
        }
    }

    initializeElements() {
        // ✅ FIX: Add null checks for all DOM elements
        this.dropzone = DOM.id('dropzone');
        this.fileInput = DOM.id('fileInput');
        
        if (!this.dropzone || !this.fileInput) {
            throw new Error('Required DOM elements not found. Check your HTML structure.');
        }
        
        // Progress elements with null safety
        this.uploadProgress = DOM.qs('.upload-progress');
        this.uploadSuccess = DOM.qs('.upload-success');
        this.uploadError = DOM.qs('.upload-error');
        this.dropzoneContent = DOM.qs('.dropzone-content');
        
        this.uploadFileName = DOM.id('uploadFileName');
        this.progressFill = DOM.id('progressFill');
        this.progressSteps = DOM.qsa('.step');
        
        // Action buttons
        this.viewResultsBtn = DOM.id('viewResultsBtn');
        this.uploadAnotherBtn = DOM.id('uploadAnotherBtn');
        this.retryBtn = DOM.id('retryBtn');
        
        // Messages
        this.successMessage = DOM.id('successMessage');
        this.errorMessage = DOM.id('errorMessage');

        this.recentList = DOM.id('recentList'); // <-- ADDED
        
        console.log('✅ DOM elements initialized');
    }

    setupEventListeners() {
        // ✅ FIX: Add null checks before adding event listeners
        if (this.dropzone) {
            this.dropzone.addEventListener('click', () => this.triggerFileSelect());
            this.dropzone.addEventListener('dragover', (e) => this.handleDragOver(e));
            this.dropzone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            this.dropzone.addEventListener('drop', (e) => this.handleDrop(e));
        }
        
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // ✅ FIX: Use optional chaining for button event listeners
        this.viewResultsBtn?.addEventListener('click', () => this.viewResults());
        this.uploadAnotherBtn?.addEventListener('click', () => this.resetUpload());
        this.retryBtn?.addEventListener('click', () => this.retryUpload());
        
        console.log('✅ Event listeners attached');
    }

    // ✅ ENHANCED: Complete file processing with better error handling
    async processFile(file) {
        try {
            console.log(`🚀 Processing: ${file.name}`);
            this.validateFile(file);
            this.currentFile = file;
            this.showProcessingState(file.name);

            // Step 1: Upload file and get OCR
            console.log('📤 Step 1: Uploading for OCR...');
            this.updateProgress(5, 'upload');
            
            const uploadResponse = await this.uploadAPI.uploadDocument(file, (percent) => {
                console.log(`Upload: ${percent.toFixed(1)}%`);
                this.updateProgress(5 + (percent * 0.25)); // 5-30%
            });

            console.log('📄 Upload response:', uploadResponse);
            if (!uploadResponse?.ocr?.full_text) {
                throw new Error('OCR failed - no text extracted from document');
            }

            this.updateProgress(35, 'ocr');
            this.setActiveStep('ocr', true);
            console.log(`✅ OCR: ${uploadResponse.ocr.full_text.length} chars extracted`);

            // Step 2: Analyze document using AnalysisAPI
            console.log('🧠 Step 2: Starting analysis...');
            this.updateProgress(40, 'analyze');
            
            const analysisResponse = await this.analysisAPI.analyzeDocument({
                ocr: uploadResponse.ocr,
                top_k: 5
            });

            console.log('📊 Analysis response:', analysisResponse);
            if (!analysisResponse) {
                throw new Error('Analysis failed - empty response from server');
            }

            this.updateProgress(90);

            // ✅ Store complete results with validation
            const completeResults = {
                filename: file.name,
                timestamp: new Date().toISOString(),
                upload_id: Date.now().toString(),
                ocr: uploadResponse.ocr,
                analysis: analysisResponse
            };

            console.log('💾 Storing results:', {
                filename: completeResults.filename,
                clauses: analysisResponse.clauses?.length || 0,
                risks: analysisResponse.risks?.length || 0,
                summary: analysisResponse.summary_200w ? 'Present' : 'Missing'
            });

            localStorage.setItem('latest_analysis', JSON.stringify(completeResults));
            this.saveToHistory(completeResults);
            this.renderRecentUploads();

            this.updateProgress(100);
            this.setActiveStep('analyze', true);
            this.showSuccessState(analysisResponse);

            console.log('✅ Processing complete - ready to navigate');
        } catch (error) {
            console.error('❌ Processing failed:', error);
            this.showErrorState(error.message || 'An unexpected error occurred during processing');
        }
    }

    saveToHistory(newResult) {
        let history = [];
        try {
            const storedHistory = localStorage.getItem('analysis_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }
        } catch (e) {
            console.error("Failed to parse analysis history", e);
            history = [];
        }

        // Add new result to the top
        history.unshift(newResult);

        // Keep only the last 5
        history = history.slice(0, 5);

        localStorage.setItem('analysis_history', JSON.stringify(history));
    }

    renderRecentUploads() {
        if (!this.recentList) return;

        let history = [];
        try {
            const storedHistory = localStorage.getItem('analysis_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }
        } catch (e) {
            console.error("Failed to parse analysis history", e);
            history = [];
        }

        if (history.length === 0) {
            this.recentList.innerHTML = `
                <div class="no-recent-message" style="text-align: center; padding: 2rem; color: #64748b;">
                    <i class="fas fa-inbox" style="font-size: 2rem; margin-bottom: 1rem; display: block;"></i>
                    <p>No recent documents yet. Upload your first document to get started!</p>
                </div>
            `;
            return;
        }

        this.recentList.innerHTML = ''; // Clear existing
        history.forEach(item => {
            const itemEl = document.createElement('div');
            itemEl.className = 'recent-item';
            itemEl.innerHTML = `
                <div class="recent-icon"><i class="fas fa-file-alt"></i></div>
                <div class="recent-details">
                    <div class="recent-filename">${item.filename}</div>
                    <div class="recent-timestamp">${new Date(item.timestamp).toLocaleString()}</div>
                </div>
                <div class="recent-actions">
                    <button class="btn btn-sm btn-outline view-results-btn" data-filename="${item.filename}">View</button>
                    <button class="btn btn-sm btn-danger delete-results-btn" data-filename="${item.filename}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            this.recentList.appendChild(itemEl);
        });

        // Add event listeners for the new buttons
        this.recentList.querySelectorAll('.view-results-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const filename = e.currentTarget.dataset.filename;
                const historyItem = history.find(item => item.filename === filename);
                if (historyItem) {
                    localStorage.setItem('latest_analysis', JSON.stringify(historyItem));
                    window.location.href = 'results.html';
                }
            });
        });

        // Add delete functionality
        this.recentList.querySelectorAll('.delete-results-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const filename = e.currentTarget.dataset.filename;
                this.deleteRecentDocument(filename);
            });
        });
    }

    deleteRecentDocument(filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
            return;
        }

        try {
            // Get current history
            let history = [];
            const storedHistory = localStorage.getItem('analysis_history');
            if (storedHistory) {
                history = JSON.parse(storedHistory);
            }

            // Remove the document from history
            const updatedHistory = history.filter(item => item.filename !== filename);
            localStorage.setItem('analysis_history', JSON.stringify(updatedHistory));

            // Also remove from latest_analysis if it's the same file
            const latestAnalysis = localStorage.getItem('latest_analysis');
            if (latestAnalysis) {
                const latest = JSON.parse(latestAnalysis);
                if (latest.filename === filename) {
                    localStorage.removeItem('latest_analysis');
                }
            }

            // Re-render the recent uploads
            this.renderRecentUploads();

            console.log(`✅ Deleted document: ${filename}`);
        } catch (error) {
            console.error('❌ Failed to delete document:', error);
            alert('Failed to delete document. Please try again.');
        }
    }

    validateFile(file) {
        const allowedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'image/png',
            'image/jpeg'
        ];
        
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Unsupported file type. Please upload PDF, DOCX, PNG, or JPEG files.');
        }

        const maxSize = 20 * 1024 * 1024; // 20MB
        if (file.size > maxSize) {
            throw new Error('File too large. Maximum size is 20MB.');
        }

        if (file.size === 0) {
            throw new Error('File is empty. Please select a valid document.');
        }
    }

    // ✅ ENHANCED: Success state with better data handling
    showSuccessState(analysisData) {
        if (this.dropzone) {
            this.dropzone.dataset.state = 'success';
        }
        
        this.safeHide(this.dropzoneContent);
        this.safeHide(this.uploadProgress);
        this.safeHide(this.uploadError);
        this.safeShow(this.uploadSuccess);
        
        const clauseCount = analysisData.clauses?.length || 0;
        const highRiskCount = analysisData.risks?.filter(r => 
            ['red', 'orange'].includes(r.level)
        ).length || 0;
        
        if (this.successMessage) {
            this.successMessage.textContent = 
                `Analysis complete! Found ${clauseCount} clauses with ${highRiskCount} high-risk items.`;
        }
            
        console.log('🎉 Success state displayed');
        
        // ✅ AUTO-NAVIGATE AFTER 2.5 SECONDS
        setTimeout(() => {
            console.log('🎯 Auto-navigating to results...');
            window.location.href = 'results.html';
        }, 2500);
    }

    showErrorState(message) {
        if (this.dropzone) {
            this.dropzone.dataset.state = 'error';
        }
        
        this.safeHide(this.dropzoneContent);
        this.safeHide(this.uploadProgress);
        this.safeHide(this.uploadSuccess);
        this.safeShow(this.uploadError);
        
        if (this.errorMessage) {
            this.errorMessage.textContent = message || 'An error occurred. Please try again.';
        }
        
        console.log('❌ Error state:', message);
    }

    showProcessingState(fileName) {
        if (this.dropzone) {
            this.dropzone.dataset.state = 'processing';
        }
        
        this.safeHide(this.dropzoneContent);
        this.safeHide(this.uploadSuccess);
        this.safeHide(this.uploadError);
        this.safeShow(this.uploadProgress);
        
        if (this.uploadFileName) {
            this.uploadFileName.textContent = `Processing: ${fileName}`;
        }
        
        this.updateProgress(0);
        this.resetSteps();
    }

    // ✅ ENHANCED: Navigate with thorough data verification
    viewResults() {
        const stored = localStorage.getItem('latest_analysis');
        if (!stored) {
            console.error('❌ No analysis data found for navigation');
            alert('No analysis data available. Please try uploading again.');
            return;
        }
        
        try {
            const data = JSON.parse(stored);
            
            // Validate data structure
            if (!data.analysis || !data.ocr) {
                throw new Error('Invalid data structure');
            }
            
            console.log('🎯 Navigating to results with data:', {
                clauses: data.analysis?.clauses?.length || 0,
                risks: data.analysis?.risks?.length || 0,
                hasOCR: !!data.ocr?.full_text
            });
            
            // Navigate to results page
            window.location.href = 'results.html';
            
        } catch (error) {
            console.error('❌ Invalid analysis data:', error);
            alert('Analysis data is corrupted. Please upload again.');
        }
    }

    resetUpload() {
        if (this.dropzone) {
            this.dropzone.dataset.state = 'idle';
        }
        this.currentFile = null;
        
        this.safeHide(this.uploadProgress);
        this.safeHide(this.uploadSuccess);
        this.safeHide(this.uploadError);
        this.safeShow(this.dropzoneContent);
        
        if (this.fileInput) {
            this.fileInput.value = '';
        }
        
        this.updateProgress(0);
        this.resetSteps();
    }

    retryUpload() {
        if (this.currentFile) {
            this.processFile(this.currentFile);
        } else {
            this.resetUpload();
        }
    }

    // ✅ ENHANCED: Progress updates with null safety
    updateProgress(percentage, step = null) {
        if (this.progressFill) {
            this.progressFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
        if (step) this.setActiveStep(step);
    }

    setActiveStep(stepName, completed = false) {
        if (!this.progressSteps) return;
        
        this.progressSteps.forEach(step => {
            const stepData = step.dataset.step;
            if (stepData === stepName) {
                step.classList.add('active');
                if (completed) {
                    step.classList.add('completed');
                    step.classList.remove('active');
                }
            } else if (!completed && step.classList.contains('active')) {
                step.classList.remove('active');
            }
        });
    }

    resetSteps() {
        if (!this.progressSteps) return;
        
        this.progressSteps.forEach(step => {
            step.classList.remove('active', 'completed');
        });
    }

    // Drag and drop handlers
    handleDragOver(e) {
        e.preventDefault();
        if (this.dropzone) {
            this.dropzone.classList.add('drag-over');
        }
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (this.dropzone && !this.dropzone.contains(e.relatedTarget)) {
            this.dropzone.classList.remove('drag-over');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        if (this.dropzone) {
            this.dropzone.classList.remove('drag-over');
        }
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    triggerFileSelect() {
        if (this.dropzone?.dataset.state === 'idle' && this.fileInput) {
            this.fileInput.click();
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    // ✅ NEW: Safe DOM manipulation helpers
    safeShow(element) {
        if (element && typeof DOM !== 'undefined' && DOM.show) {
            DOM.show(element);
        } else if (element) {
            element.style.display = 'block';
        }
    }

    safeHide(element) {
        if (element && typeof DOM !== 'undefined' && DOM.hide) {
            DOM.hide(element);
        } else if (element) {
            element.style.display = 'none';
        }
    }

    // ✅ NEW: Generic error display method
    showError(message) {
        console.error('UploadPage Error:', message);
        if (this.errorMessage) {
            this.errorMessage.textContent = message;
        }
        alert(message); // Fallback for critical errors
    }
}

// ✅ ENHANCED: Initialize with comprehensive error handling
document.addEventListener('DOMContentLoaded', () => {
    try {
        window.uploadPage = new UploadPage();
        console.log('✅ Upload page initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize upload page:', error);
        
        // Show user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
            background: #fee; border: 1px solid #fcc; color: #c33;
            padding: 15px 20px; border-radius: 8px; z-index: 1000;
            font-family: system-ui, -apple-system, sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;
        errorDiv.innerHTML = `
            <strong>Upload page failed to load</strong><br>
            ${error.message}<br>
            <small>Please refresh the page and try again.</small>
        `;
        document.body.appendChild(errorDiv);
        
        // Auto-remove error after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 10000);
    }
});