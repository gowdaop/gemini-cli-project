// Router - frontend/static/js/router.js

class Router {
  constructor() {
    this.routes = new Map();
    this.currentRoute = null;
    this.basePath = '';
    this.init();
  }

  init() {
    // Setup event listeners
    window.addEventListener('popstate', () => {
      this.handleRouteChange();
    });

    // Handle initial route
    this.handleRouteChange();
  }

  // Define routes
  defineRoutes() {
    this.addRoute('/', {
      template: 'index.html',
      title: 'Home - Legal Document Simplifier',
      component: null
    });

    this.addRoute('/upload', {
      template: 'pages/upload.html',
      title: 'Upload Documents - Legal Document Simplifier',
      component: 'UploadPage'
    });

    this.addRoute('/results', {
      template: 'pages/results.html',
      title: 'Analysis Results - Legal Document Simplifier',
      component: 'ResultsPage'
    });

    this.addRoute('/chat', {
      template: 'pages/chat.html',
      title: 'AI Assistant - Legal Document Simplifier',
      component: 'ChatPage'
    });
  }

  addRoute(path, config) {
    this.routes.set(path, config);
  }

  getCurrentPath() {
    const path = window.location.pathname;
    
    // Normalize path based on file structure
    if (path.includes('index.html') || path.endsWith('/')) {
      return '/';
    }
    if (path.includes('upload.html')) {
      return '/upload';
    }
    if (path.includes('results.html')) {
      return '/results';
    }
    if (path.includes('chat.html')) {
      return '/chat';
    }
    
    return '/';
  }

  handleRouteChange() {
    const currentPath = this.getCurrentPath();
    this.currentRoute = currentPath;
    
    // Update document title
    const route = this.routes.get(currentPath);
    if (route && route.title) {
      document.title = route.title;
    }
    
    // Update active navigation
    this.updateActiveNavigation(currentPath);
    
    // Initialize page component
    this.initializePageComponent(currentPath);
    
    // Trigger route change event
    this.triggerRouteChangeEvent(currentPath);
  }

  updateActiveNavigation(currentPath) {
    // Update header navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
      link.classList.remove('active');
      
      const href = link.getAttribute('href');
      let linkPath = '/';
      
      if (href.includes('upload')) linkPath = '/upload';
      else if (href.includes('results')) linkPath = '/results';
      else if (href.includes('chat')) linkPath = '/chat';
      
      if (linkPath === currentPath) {
        link.classList.add('active');
      }
    });
    
    // Update header component if available
    if (window.headerComponent) {
      const pageMap = {
        '/': 'home',
        '/upload': 'upload',
        '/results': 'results',
        '/chat': 'chat'
      };
      window.headerComponent.setActivePage(pageMap[currentPath] || 'home');
    }
  }

  initializePageComponent(currentPath) {
    const route = this.routes.get(currentPath);
    
    if (route && route.component && window[route.component]) {
      // Initialize page component if available
      try {
        new window[route.component]();
      } catch (error) {
        console.warn(`Failed to initialize component ${route.component}:`, error);
      }
    }
    
    // Page-specific initialization
    switch (currentPath) {
      case '/':
        this.initHomePage();
        break;
      case '/upload':
        this.initUploadPage();
        break;
      case '/results':
        this.initResultsPage();
        break;
      case '/chat':
        this.initChatPage();
        break;
    }
  }

  initHomePage() {
    // Initialize home page functionality
    this.setupHomePageScrolling();
    this.setupDashboardNavigation();
  }

  initUploadPage() {
    // Initialize upload page
    if (window.DropzoneComponent) {
      new window.DropzoneComponent();
    }
    this.setupUploadHandlers();
  }

  initResultsPage() {
    // Initialize results page
    if (window.TableComponent) {
      new window.TableComponent();
    }
    if (window.PDFViewerComponent) {
      new window.PDFViewerComponent();
    }
    this.loadResultsData();
  }

  initChatPage() {
    // Initialize chat page
    if (window.ChatComponent) {
      new window.ChatComponent();
    }
    this.setupChatHandlers();
  }

  setupHomePageScrolling() {
    // Smooth scroll to sections
    window.scrollToSection = (sectionId) => {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    };
  }

  setupDashboardNavigation() {
    // Handle dashboard card clicks
    window.navigateToPage = (page) => {
      const routes = {
        'upload': '../templates/pages/upload.html',
        'results': '../templates/pages/results.html',
        'chat': '../templates/pages/chat.html'
      };
      
      if (routes[page]) {
        this.navigateTo(routes[page]);
      }
    };
  }

  setupUploadHandlers() {
    // File upload specific handlers
    console.log('Setting up upload page handlers...');
  }

  setupChatHandlers() {
    // Chat specific handlers
    console.log('Setting up chat page handlers...');
  }

  loadResultsData() {
    // Load and display results data
    console.log('Loading results data...');
  }

  navigateTo(url, options = {}) {
    if (options.external || url.startsWith('http')) {
      // External navigation
      window.location.href = url;
    } else {
      // Internal navigation
      window.location.href = url;
    }
  }

  triggerRouteChangeEvent(path) {
    // Custom route change event
    const event = new CustomEvent('routechange', {
      detail: {
        path: path,
        route: this.routes.get(path)
      }
    });
    window.dispatchEvent(event);
  }

  // Utility methods
  getRouteParams() {
    const params = new URLSearchParams(window.location.search);
    const result = {};
    for (const [key, value] of params) {
      result[key] = value;
    }
    return result;
  }

  updateUrl(params, replace = false) {
    const url = new URL(window.location);
    
    Object.entries(params).forEach(([key, value]) => {
      if (value === null || value === undefined) {
        url.searchParams.delete(key);
      } else {
        url.searchParams.set(key, value);
      }
    });
    
    if (replace) {
      window.history.replaceState({}, '', url);
    } else {
      window.history.pushState({}, '', url);
    }
  }

  // Breadcrumb generation
  generateBreadcrumbs() {
    const path = this.currentRoute;
    const breadcrumbs = [];
    
    // Always include home
    breadcrumbs.push({
      text: 'Home',
      url: '../index.html',
      active: path === '/'
    });
    
    // Add current page if not home
    if (path !== '/') {
      const routeNames = {
        '/upload': 'Upload Documents',
        '/results': 'Analysis Results',
        '/chat': 'AI Assistant'
      };
      
      breadcrumbs.push({
        text: routeNames[path] || 'Page',
        url: null,
        active: true
      });
    }
    
    return breadcrumbs;
  }

  renderBreadcrumbs(container) {
    if (!container) return;
    
    const breadcrumbs = this.generateBreadcrumbs();
    const breadcrumbHTML = breadcrumbs.map((item, index) => {
      if (item.active) {
        return `<span class="breadcrumb-active">${item.text}</span>`;
      } else {
        return `<a href="${item.url}" class="breadcrumb-link">${item.text}</a>`;
      }
    }).join('<span class="breadcrumb-separator">></span>');
    
    container.innerHTML = `<div class="breadcrumb">${breadcrumbHTML}</div>`;
  }
}

// Initialize router when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.router = new Router();
    window.router.defineRoutes();
  });
} else {
  window.router = new Router();
  window.router.defineRoutes();
}

// Make Router available globally
window.Router = Router;