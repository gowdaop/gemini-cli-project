// Header Component - frontend/static/js/components/header.js

class HeaderComponent {
  constructor() {
    this.header = null;
    this.navToggle = null;
    this.navMenu = null;
    this.currentPage = this.getCurrentPage();
    this.init();
  }

  init() {
    this.createHeader();
    this.setupEventListeners();
    this.updateActiveNavItem();
  }

  getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('upload')) return 'upload';
    if (path.includes('results')) return 'results';
    if (path.includes('chat')) return 'chat';
    if (path.includes('index') || path.endsWith('/')) return 'home';
    return 'home';
  }

  createHeader() {
    // Check if header already exists
    let existingHeader = document.querySelector('.header');
    
    if (!existingHeader) {
      // Create header if it doesn't exist
      const headerHTML = this.getHeaderHTML();
      document.body.insertAdjacentHTML('afterbegin', headerHTML);
    }
    
    // Get header elements
    this.header = document.querySelector('.header');
    this.navToggle = document.querySelector('.nav-toggle');
    this.navMenu = document.querySelector('.nav-menu');
  }

  getHeaderHTML() {
    return `
      <header class="header">
        <nav class="nav-container">
          <div class="nav-brand">
            <i class="fas fa-balance-scale"></i>
            <span>LegalSimplifier</span>
          </div>
          <div class="nav-menu">
            <a href="../index.html" class="nav-link" data-page="home">Home</a>
            <a href="../templates/pages/upload.html" class="nav-link" data-page="upload">Upload</a>
            <a href="../templates/pages/results.html" class="nav-link" data-page="results">Results</a>
            <a href="../templates/pages/chat.html" class="nav-link" data-page="chat">Chat</a>
          </div>
          <div class="nav-actions">
            <button class="btn btn-outline" id="loginBtn">Login</button>
            <button class="btn btn-primary" id="signupBtn">Get Started</button>
          </div>
          <button class="nav-toggle" id="navToggle">
            <span class="nav-toggle-bar"></span>
            <span class="nav-toggle-bar"></span>
            <span class="nav-toggle-bar"></span>
          </button>
        </nav>
      </header>
    `;
  }

  setupEventListeners() {
    // Mobile menu toggle
    if (this.navToggle && this.navMenu) {
      this.navToggle.addEventListener('click', () => {
        this.toggleMobileMenu();
      });
    }

    // Navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        this.handleNavClick(e);
      });
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.header.contains(e.target)) {
        this.closeMobileMenu();
      }
    });

    // Login button
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
      loginBtn.addEventListener('click', () => {
        this.handleLogin();
      });
    }

    // Signup button
    const signupBtn = document.getElementById('signupBtn');
    if (signupBtn) {
      signupBtn.addEventListener('click', () => {
        this.handleSignup();
      });
    }

    // Brand logo click
    const navBrand = document.querySelector('.nav-brand');
    if (navBrand) {
      navBrand.addEventListener('click', () => {
        window.location.href = '../index.html';
      });
      navBrand.style.cursor = 'pointer';
    }

    // Scroll behavior
    this.setupScrollBehavior();
  }

  toggleMobileMenu() {
    this.navMenu.classList.toggle('active');
    this.navToggle.classList.toggle('active');
    
    // Prevent body scroll when menu is open
    if (this.navMenu.classList.contains('active')) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
  }

  closeMobileMenu() {
    this.navMenu.classList.remove('active');
    this.navToggle.classList.remove('active');
    document.body.style.overflow = '';
  }

  handleNavClick(e) {
    // Close mobile menu
    this.closeMobileMenu();
    
    // Optional: Add loading state
    const link = e.currentTarget;
    link.style.opacity = '0.7';
    
    setTimeout(() => {
      if (link) {
        link.style.opacity = '';
      }
    }, 200);
  }

  updateActiveNavItem() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
      link.classList.remove('active');
      const linkPage = link.getAttribute('data-page');
      if (linkPage === this.currentPage) {
        link.classList.add('active');
      }
    });
  }

  handleLogin() {
    // For now, just show a placeholder
    if (window.showToast) {
      window.showToast('Login functionality coming soon!', 'info');
    } else {
      alert('Login functionality coming soon!');
    }
    
    // TODO: Implement actual login functionality
    // This could open a modal or redirect to a login page
  }

  handleSignup() {
    // For now, just show a placeholder
    if (window.showToast) {
      window.showToast('Sign up functionality coming soon!', 'info');
    } else {
      alert('Sign up functionality coming soon!');
    }
    
    // TODO: Implement actual signup functionality
  }

  setupScrollBehavior() {
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', FunctionUtils.throttle(() => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      
      if (scrollTop > lastScrollTop && scrollTop > 100) {
        // Scrolling down - hide header
        this.header.classList.add('header-hidden');
      } else {
        // Scrolling up - show header
        this.header.classList.remove('header-hidden');
      }
      
      // Add background blur when scrolled
      if (scrollTop > 50) {
        this.header.classList.add('header-scrolled');
      } else {
        this.header.classList.remove('header-scrolled');
      }
      
      lastScrollTop = scrollTop;
    }, 100));
  }

  // Public methods
  setActivePage(page) {
    this.currentPage = page;
    this.updateActiveNavItem();
  }

  showNotification(message, type = 'info') {
    // Create a small notification in the header
    const notification = DOM.create('div', {
      className: `header-notification ${type}`,
      innerHTML: `
        <span>${message}</span>
        <button class="notification-close">&times;</button>
      `
    });

    this.header.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 3000);

    // Close button
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
      notification.remove();
    });
  }
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    if (!window.headerComponent) {
      window.headerComponent = new HeaderComponent();
    }
  });
} else {
  if (!window.headerComponent) {
    window.headerComponent = new HeaderComponent();
  }
}

// Make component available globally
window.HeaderComponent = HeaderComponent;