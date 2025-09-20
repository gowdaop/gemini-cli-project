/*  -------------------------------
    Results Page Controller
    ------------------------------- */

class ResultsPage {
  constructor() {
    this.analysisData   = null;       // full JSON from backend
    this.filteredClauses = [];        // current clause view
    this.charts         = {};         // Chart-JS instances
    this.init();                      // bootstrap
  }

  /* ---------- 1.  BOOTSTRAP ---------- */
  async init() {
    try {
      console.log('🎯 Loading results page…');

      await this.loadRealAnalysisData();
      if (!this.analysisData) throw new Error('No analysis data available');

      console.log('📊 Real data loaded:', {
        filename : this.analysisData.filename,
        clauses  : this.analysisData.analysis?.clauses?.length || 0,
        risks    : this.analysisData.analysis?.risks?.length   || 0,
        summary  : !!this.analysisData.analysis?.summary_200w
      });

      this.populateDocumentInfo();
      this.populateOverviewCards();
      this.initializeCharts();
      this.renderClauses();
      this.showRecommendations();
      this.setupEventListeners();
      this.setupModal();

      console.log('✅ Results page fully rendered');
    } catch (err) {
      console.error('❌ Results page error:', err);
      this.showError(err.message || 'Unexpected error');
    }
  }

  /* ---------- 2.  DATA LOADING ---------- */
  async loadRealAnalysisData() {
    const stored = localStorage.getItem('latest_analysis');
    if (!stored) {
      console.warn('⚠️ No stored analysis – redirecting to upload');
      window.location.href = 'upload.html';
      return;
    }

    try {
      this.analysisData    = JSON.parse(stored);
      this.filteredClauses = this.analysisData.analysis?.clauses || [];
      console.log('✅ Analysis data parsed from storage');
    } catch (err) {
      console.error('❌ Failed to parse analysis JSON:', err);
      throw err;
    }
  }

  /* ---------- 3.  DOM POPULATION ---------- */
  populateDocumentInfo() {
    const { filename = 'Unknown document', timestamp = Date.now() } = this.analysisData;
    const titleEl  = DOM.id('documentTitle');
    const timeEl   = DOM.id('analysisTime');
    const pagesEl  = DOM.id('documentPages');

    if (titleEl) titleEl.textContent = filename;
    if (timeEl)  timeEl.textContent  = DateUtils.relative(new Date(timestamp));

    // pages from OCR blocks
    const blocks   = this.analysisData.ocr?.blocks || [];
    const maxPage  = Math.max(...blocks.map(b => b.span?.page || 1), 1);
    if (pagesEl) pagesEl.textContent = `${maxPage} page${maxPage > 1 ? 's' : ''}`;
  }

  populateOverviewCards() {
    const clauses = this.analysisData.analysis?.clauses || [];
    const risks   = this.analysisData.analysis?.risks   || [];

    const riskCounts     = this.calculateRiskDistribution(risks);
    const highRiskCount  = riskCounts.red + riskCounts.orange;
    const averageRisk    = this.calculateAverageRiskScore(risks);

    // safe writes
    const safeSet = (id, txt) => { const el = DOM.id(id); if (el) el.textContent = txt; };
    safeSet('totalClauses'  , clauses.length);
    safeSet('highRiskCount' , highRiskCount);
    safeSet('averageRiskScore', `${(averageRisk * 10).toFixed(1)}/10`);
    safeSet('redCount'    , riskCounts.red);
    safeSet('orangeCount' , riskCounts.orange);
    safeSet('yellowCount' , riskCounts.yellow);
    safeSet('greenCount'  , riskCounts.white);

    // overall risk badge
    this.setOverallRiskLevel(riskCounts);

    // summary with enhanced formatting
    this.populateSummary();
  }

  populateSummary() {
    const summaryEl = DOM.id('documentSummary');
    if (!summaryEl) return;

    const summaryText = this.analysisData.analysis?.summary_200w || 'No summary available.';
    
    // Parse the summary to separate SUMMARY and RECOMMENDATIONS sections
    const { summarySection, recommendationsSection } = this.parseSummarySections(summaryText);
    
    // Create enhanced HTML structure
    summaryEl.innerHTML = `
      <div class="summary-content">
        ${summarySection ? `
          <div class="summary-section">
            <div class="section-header">
              <i class="fas fa-file-alt"></i>
              <h4>Document Summary</h4>
            </div>
            <div class="section-content">
              ${this.formatSummaryText(summarySection)}
            </div>
          </div>
        ` : ''}
        
        ${recommendationsSection ? `
          <div class="recommendations-section">
            <div class="section-header">
              <i class="fas fa-lightbulb"></i>
              <h4>Key Recommendations</h4>
            </div>
            <div class="section-content">
              ${this.formatRecommendationsText(recommendationsSection)}
            </div>
          </div>
        ` : ''}
        
        ${!summarySection && !recommendationsSection ? `
          <div class="summary-fallback">
            <p>${summaryText}</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  parseSummarySections(text) {
    if (!text) return { summarySection: null, recommendationsSection: null };

    // Look for explicit section markers
    const summaryMatch = text.match(/SUMMARY[:\s]*(.*?)(?=RECOMMENDATIONS|$)/is);
    const recommendationsMatch = text.match(/RECOMMENDATIONS[:\s]*(.*?)$/is);

    if (summaryMatch && recommendationsMatch) {
      return {
        summarySection: summaryMatch[1].trim(),
        recommendationsSection: recommendationsMatch[1].trim()
      };
    }

    // If no explicit markers, try to split on common patterns
    const lines = text.split('\n');
    let summaryLines = [];
    let recommendationsLines = [];
    let currentSection = 'summary';

    for (const line of lines) {
      const trimmedLine = line.trim();
      
      if (trimmedLine.match(/^(recommendations?|key recommendations?|suggestions?)/i)) {
        currentSection = 'recommendations';
        continue;
      }
      
      if (trimmedLine) {
        if (currentSection === 'summary') {
          summaryLines.push(trimmedLine);
        } else {
          recommendationsLines.push(trimmedLine);
        }
      }
    }

    return {
      summarySection: summaryLines.length > 0 ? summaryLines.join(' ') : null,
      recommendationsSection: recommendationsLines.length > 0 ? recommendationsLines.join(' ') : null
    };
  }

  formatSummaryText(text) {
    if (!text) return '';
    
    // Clean up the text
    let formatted = text.trim();
    
    // Add paragraph breaks for better readability
    formatted = formatted.replace(/([.!?])\s+(?=[A-Z])/g, '$1\n\n');
    
    // Split into paragraphs and wrap in <p> tags
    const paragraphs = formatted.split('\n\n').filter(p => p.trim());
    
    return paragraphs.map(p => `<p>${p.trim()}</p>`).join('');
  }

  formatRecommendationsText(text) {
    if (!text) return '';
    
    // Clean up the text
    let formatted = text.trim();
    
    // Check if it's already a numbered list
    if (formatted.match(/^\d+\./)) {
      // Split by numbered items and format as list
      const items = formatted.split(/(?=^\d+\.)/m).filter(item => item.trim());
      
      return `
        <ol class="recommendations-list">
          ${items.map(item => {
            const cleaned = item.replace(/^\d+\.\s*/, '').trim();
            return `<li>${cleaned}</li>`;
          }).join('')}
        </ol>
      `;
    } else {
      // Try to split by common list patterns
      const lines = formatted.split('\n').filter(line => line.trim());
      
      if (lines.length > 1) {
        return `
          <ul class="recommendations-list">
            ${lines.map(line => `<li>${line.trim()}</li>`).join('')}
          </ul>
        `;
      } else {
        return `<p>${formatted}</p>`;
      }
    }
  }

  /* ---------- 4.  CHARTS ---------- */
  initializeCharts() {
    this.createRiskChart();
    this.createClauseChart();
  }

  createRiskChart() {
    const ctx = DOM.id('riskChart');
    if (!ctx) return;

    const risks        = this.analysisData.analysis?.risks || [];
    const distribution = this.calculateRiskDistribution(risks);

    // destroy previous
    if (this.charts.risk) this.charts.risk.destroy();

    this.charts.risk = new Chart(ctx, {
      type : 'doughnut',
      data : {
        labels     : ['High Risk', 'Medium-High', 'Medium', 'Low Risk'],
        datasets   : [{
          data            : [distribution.red, distribution.orange, distribution.yellow, distribution.white],
          backgroundColor : ['#ef4444', '#f59e0b', '#fcd34d', '#22c55e'],
          borderWidth     : 0
        }]
      },
      options : {
        responsive : true,
        maintainAspectRatio : false,
        plugins : { legend : { display : false } }
      }
    });
  }

  createClauseChart() {
    const ctx = DOM.id('clauseChart');
    if (!ctx) return;

    const clauses      = this.analysisData.analysis?.clauses || [];
    const distribution = {};
    clauses.forEach(c => {
      const t = c.tag || 'other';
      distribution[t] = (distribution[t] || 0) + 1;
    });

    // destroy previous
    if (this.charts.clause) this.charts.clause.destroy();

    const labels = Object.keys(distribution);
    const data   = Object.values(distribution);
    const colors = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#818cf8'];

    this.charts.clause = new Chart(ctx, {
      type    : 'bar',
      data    : {
        labels   : labels.map(this.formatClauseType),
        datasets : [{ data, backgroundColor : colors.slice(0, data.length) }]
      },
      options : {
        responsive : true,
        maintainAspectRatio : false,
        scales  : { y : { beginAtZero : true } },
        plugins : { legend : { display : false } }
      }
    });
  }

  /* ---------- 5.  CLAUSE LIST ---------- */
  renderClauses() {
    const container = DOM.id('clausesContainer');
    if (!container) return;

    container.innerHTML = '';

    if (this.filteredClauses.length === 0) {
      container.innerHTML = `
        <div style="text-align:center;padding:2rem;color:#64748b">
          <p>No clauses found with current filters.</p>
        </div>`;
      return;
    }

    this.filteredClauses.forEach(clause => {
      const risk   = this.findRiskForClause(clause.id);
      const element = this.createClauseElement(clause, risk);
      container.appendChild(element);
    });
  }

showRecommendations() {
  // pick the single riskiest clause; tweak if you prefer
  const topRisk = (this.analysisData.analysis?.risks || [])
                    .sort((a, b) => b.score - a.score)[0];

  RenderUtils.list(
    DOM.id('actionItems'),
    topRisk?.recommendations || []
  );
}

  /* ---------- 6.  HELPERS ---------- */
  calculateRiskDistribution(risks) {
    const dist = { red:0, orange:0, yellow:0, white:0 };
    risks.forEach(r => {
      const lvl = r.level || 'white';
      if (lvl in dist) dist[lvl] += 1;
    });
    return dist;
  }

  calculateAverageRiskScore(risks) {
    if (risks.length === 0) return 0;
    return risks.reduce((s, r) => s + (r.score || 0), 0) / risks.length;
  }

  setOverallRiskLevel(counts) {
    const total = counts.red + counts.orange + counts.yellow + counts.white;
    const ratio = total ? (counts.red * 4 + counts.orange * 3 + counts.yellow * 2) / (4 * total) : 0;
    const el    = DOM.id('overallRiskLevel');
    const desc  = DOM.id('overallRiskDesc');
    const icon  = DOM.id('overallRiskIcon');

    let text = 'Low', css = 'low', detail = 'Looks safe';
    if (ratio > 0.6) { text='High'; css='high'; detail='Needs immediate attention'; }
    else if (ratio > 0.35) { text='Medium-High'; css='medium'; detail='Requires careful review'; }
    else if (ratio > 0.15) { text='Medium'; css='medium'; detail='Contains some risk'; }

    if (el)  el.textContent  = text;
    if (desc) desc.textContent = detail;
    if (icon) {
      icon.classList.remove('low','medium','high');
      icon.classList.add(css);
    }
  }

  findRiskForClause(id) {
    const risks = this.analysisData.analysis?.risks || [];
    return risks.find(r => r.clause_id === id) || { level:'white', score:0, rationale:'N/A' };
  }

  formatClauseType(type) {
    if (!type) return 'Other';
    return type.split('_')
               .map(w => w.charAt(0).toUpperCase() + w.slice(1))
               .join(' ');
  }

  createClauseElement(clause, risk) {
    const div = document.createElement('div');
    div.className = 'clause-item';
    div.innerHTML = `
      <div class="clause-header risk-${risk.level}">
        <div class="clause-info">
          <div class="clause-title">
            <span class="clause-type">${this.formatClauseType(clause.tag)}</span>
            <span class="clause-id">${clause.id}</span>
          </div>
          <div class="clause-preview">
            ${StringUtils.truncate(clause.text, 150)}
          </div>
        </div>
        <div class="clause-meta">
          <span class="risk-badge ${risk.level}">${risk.level.toUpperCase()}</span>
          <span class="risk-score">Score: ${(risk.score || 0).toFixed(2)}</span>
        </div>
      </div>`;
    // open modal on click
    div.addEventListener('click', () => this.showClauseModal(clause, risk));
    return div;
  }

  showToast(message, type = 'info') {
    const toastContainer = DOM.id('toastContainer') || document.body;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
      </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 3000);
  }

  /* ---------- 7.  MODAL & EVENTS ---------- */
  setupModal() { /* implement open/close logic here */ }
  showClauseModal(clause, risk) { /* fill and display modal */ }

  setupEventListeners() {
    // copy summary
    const copyBtn = DOM.id('copySummary');
    if (copyBtn) copyBtn.addEventListener('click', () => {
      const summaryText = this.analysisData.analysis?.summary_200w || '';
      navigator.clipboard.writeText(summaryText).then(() => {
        // Show a better notification
        this.showToast('Summary copied to clipboard', 'success');
      }).catch(() => {
        this.showToast('Failed to copy summary', 'error');
      });
    });

    // filters
    const riskFilter  = DOM.id('riskFilter');
    const typeFilter  = DOM.id('typeFilter');
    const applyFilter = () => {
      const riskVal = riskFilter?.value || 'all';
      const typeVal = typeFilter?.value || 'all';
      this.filteredClauses = this.analysisData.analysis.clauses.filter(c => {
        const risk  = this.findRiskForClause(c.id);
        return (riskVal === 'all'  || risk.level === riskVal) &&
               (typeVal === 'all'  || c.tag === typeVal);
      });
      this.renderClauses();
    };
    riskFilter?.addEventListener('change', applyFilter);
    typeFilter?.addEventListener('change', applyFilter);

    // chat button
    const chatBtn = DOM.id('chatBtn');
    if (chatBtn) {
      chatBtn.addEventListener('click', () => {
        if (this.analysisData && this.analysisData.filename) {
          // Pass the filename to the chat page
          window.location.href = `chat.html?filename=${encodeURIComponent(this.analysisData.filename)}`;
        } else {
          // Fallback for safety
          window.location.href = 'chat.html';
        }
      });
    }
  }

  /* ---------- 8.  ERROR UI ---------- */
  showError(msg) {
    alert(msg);          // simple fallback – replace with fancy toast if you like
  }
}

/* ---------- 9.  BOOT ---------- */
document.addEventListener('DOMContentLoaded', () => {
  window.resultsPage = new ResultsPage();
});
