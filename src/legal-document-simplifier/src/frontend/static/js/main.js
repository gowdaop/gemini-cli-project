// Main JavaScript - frontend/static/js/main.js

class LegalSimplifier {
  constructor() {
    this.currentPage = this.detectCurrentPage();
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.loadComponents();
    this.initializeCurrentPage();
  }

  detectCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('upload')) return 'upload';
    if (path.includes('results')) return 'results';
    if (path.includes('chat')) return 'chat';
    return 'home';
  }

  setupEventListeners() {
    // Global event listeners
    document.addEventListener('DOMContentLoaded', () => {
      this.handleDOMReady();
    });

    // Navigation handlers
    this.setupNavigationListeners();

    // Mobile menu toggle
    this.setupMobileMenu();

    // Loading states
    this.setupLoadingStates();
  }

  handleDOMReady() {
    // Initialize tooltips
    this.initTooltips();
    
    // Initialize smooth scrolling
    this.initSmoothScrolling();
    
    // Initialize animations
    this.initAnimations();
  }

  setupNavigationListeners() {
    // Header navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        this.handleNavigation(e);
      });
    });

    // Dashboard card navigation
    const dashboardCards = document.querySelectorAll('.dashboard-card');
    dashboardCards.forEach(card => {
      card.addEventListener('click', (e) => {
        this.handleDashboardNavigation(e);
      });
    });
  }

  setupMobileMenu() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.querySelector('.nav-menu');

    if (navToggle && navMenu) {
      navToggle.addEventListener('click', () => {
        navMenu.classList.toggle('active');
        navToggle.classList.toggle('active');
      });

      // Close menu on outside click
      document.addEventListener('click', (e) => {
        if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
          navMenu.classList.remove('active');
          navToggle.classList.remove('active');
        }
      });
    }
  }

  setupLoadingStates() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    
    window.showLoading = (message = 'Processing...') => {
      if (loadingOverlay) {
        const loadingText = loadingOverlay.querySelector('p');
        if (loadingText) loadingText.textContent = message;
        loadingOverlay.classList.add('active');
      }
    };

    window.hideLoading = () => {
      if (loadingOverlay) {
        loadingOverlay.classList.remove('active');
      }
    };
  }

  loadComponents() {
    // Load header component
    if (window.HeaderComponent) {
      new HeaderComponent();
    }

    // Load page-specific components
    switch (this.currentPage) {
      case 'upload':
        this.loadUploadComponents();
        break;
      case 'results':
        this.loadResultsComponents();
        break;
      case 'chat':
        this.loadChatComponents();
        break;
    }
  }

  loadUploadComponents() {
    if (window.DropzoneComponent) {
      new DropzoneComponent();
    }
  }

  loadResultsComponents() {
    if (window.TableComponent) {
      new TableComponent();
    }
    if (window.PDFViewerComponent) {
      new PDFViewerComponent();
    }
  }

  loadChatComponents() {
    if (window.ChatComponent) {
      new ChatComponent();
    }
  }

  initializeCurrentPage() {
    // Page-specific initialization
    switch (this.currentPage) {
      case 'home':
        this.initHomePage();
        break;
      case 'upload':
        this.initUploadPage();
        break;
      case 'results':
        this.initResultsPage();
        break;
      case 'chat':
        this.initChatPage();
        break;
    }
  }

  initHomePage() {
    // Scroll to section functionality
    window.scrollToSection = (sectionId) => {
      const section = document.getElementById(sectionId);
      if (section) {
        section.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }
    };

    // Dashboard navigation
    window.navigateToPage = (page) => {
      const pages = {
        'upload': '../templates/pages/upload.html',
        'results': '../templates/pages/results.html',
        'chat': '../templates/pages/chat.html'
      };
      
      if (pages[page]) {
        window.location.href = pages[page];
      }
    };
  }

  initUploadPage() {
    console.log('Initializing upload page...');
  }

  initResultsPage() {
    console.log('Initializing results page...');
  }

  initChatPage() {
    console.log('Initializing chat page...');
  }

    handleNavigation(e) {
    e.preventDefault();
    const href = e.currentTarget.getAttribute('href');
    if (href && href !== '#') {
      window.location.href = href;
    }
  }

  handleDashboardNavigation(e) {
    const card = e.currentTarget;
    const onclick = card.getAttribute('onclick');
    if (onclick) {
      // Extract page name from onclick attribute
      const match = onclick.match(/navigateToPage\('(\w+)'\)/);
      if (match) {
        window.navigateToPage(match[1]);
      }
    }
  }

  initTooltips() {
    // Simple tooltip implementation
    const elementsWithTooltips = document.querySelectorAll('[data-tooltip]');
    elementsWithTooltips.forEach(element => {
      element.addEventListener('mouseenter', (e) => {
        this.showTooltip(e);
      });
      element.addEventListener('mouseleave', (e) => {
        this.hideTooltip(e);
      });
    });
  }

  showTooltip(e) {
    const tooltipText = e.target.getAttribute('data-tooltip');
    if (!tooltipText) return;

    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;
    document.body.appendChild(tooltip);

    const rect = e.target.getBoundingClientRect();
    tooltip.style.position = 'absolute';
    tooltip.style.top = `${rect.bottom + 8}px`;
    tooltip.style.left = `${rect.left + (rect.width / 2)}px`;
    tooltip.style.transform = 'translateX(-50%)';
    tooltip.style.zIndex = '1000';
    tooltip.style.background = 'var(--secondary-900)';
    tooltip.style.color = 'white';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '14px';
    tooltip.style.whiteSpace = 'nowrap';

    e.target._tooltip = tooltip;
  }

  hideTooltip(e) {
    if (e.target._tooltip) {
      document.body.removeChild(e.target._tooltip);
      delete e.target._tooltip;
    }
  }

  initSmoothScrolling() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth'
          });
        }
      });
    });
  }

  initAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    }, observerOptions);

    // Observe elements that should fade in
    document.querySelectorAll('.feature-card, .dashboard-card').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      observer.observe(el);
    });
  }

  // Toast notification system
  showToast(message, type = 'info', duration = 5000) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <div class="toast-content">
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.parentElement.remove()">
          <i class="fas fa-times"></i>
        </button>
      </div>
    `;

    toastContainer.appendChild(toast);

    // Trigger show animation
    setTimeout(() => {
      toast.classList.add('show');
    }, 100);

    // Auto remove
    setTimeout(() => {
      if (toast.parentNode) {
        toast.classList.remove('show');
        setTimeout(() => {
          if (toast.parentNode) {
            toast.remove();
          }
        }, 300);
      }
    }, duration);
  }

  // Error handling
  handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    this.showToast(`An error occurred: ${error.message}`, 'error');
  }
}

// Toast styles
const toastStyles = `
  .toast {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--primary-500);
    min-width: 300px;
    max-width: 400px;
    transform: translateX(100%);
    opacity: 0;
    transition: all 0.3s ease;
    margin-bottom: 8px;
  }
  
  .toast.show {
    transform: translateX(0);
    opacity: 1;
  }
  
  .toast.success { border-left-color: var(--success-500); }
  .toast.error { border-left-color: var(--error-500); }
  .toast.warning { border-left-color: var(--warning-500); }
  
  .toast-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  
  .toast-close {
    background: none;
    border: none;
    color: var(--secondary-400);
    cursor: pointer;
    padding: 4px;
  }
  
  .toast-close:hover {
    color: var(--secondary-600);
  }
`;

// Inject toast styles
const styleSheet = document.createElement('style');
styleSheet.textContent = toastStyles;
document.head.appendChild(styleSheet);

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.app = new LegalSimplifier();
  });
} else {
  window.app = new LegalSimplifier();
}

// Global utility functions
window.showToast = (message, type, duration) => {
  if (window.app) {
    window.app.showToast(message, type, duration);
  }
};

window.showError = (message) => {
  window.showToast(message, 'error');
};

window.showSuccess = (message) => {
  window.showToast(message, 'success');
};