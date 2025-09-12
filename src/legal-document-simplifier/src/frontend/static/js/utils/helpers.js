// Helper Utilities - frontend/static/js/utils/helpers.js

// DOM Utilities
const DOM = {
  // Get element by ID
  id: (id) => document.getElementById(id),
  
  // Query selector
  qs: (selector) => document.querySelector(selector),
  
  // Query selector all
  qsa: (selector) => document.querySelectorAll(selector),
  
  // Create element
  create: (tag, attributes = {}, content = '') => {
    const element = document.createElement(tag);
    
    // Set attributes
    Object.entries(attributes).forEach(([key, value]) => {
      if (key === 'className') {
        element.className = value;
      } else if (key === 'innerHTML') {
        element.innerHTML = value;
      } else {
        element.setAttribute(key, value);
      }
    });
    
    // Set content
    if (content) {
      element.textContent = content;
    }
    
    return element;
  },
  
  // Add event listener
  on: (element, event, handler) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.addEventListener(event, handler);
    }
  },
  
  // Add class
  addClass: (element, className) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.classList.add(className);
    }
  },
  
  // Remove class
  removeClass: (element, className) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.classList.remove(className);
    }
  },
  
  // Toggle class
  toggleClass: (element, className) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.classList.toggle(className);
    }
  },
  
  // Show element
  show: (element) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.style.display = 'block';
    }
  },
  
  // Hide element
  hide: (element) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    if (element) {
      element.style.display = 'none';
    }
  },
  
  // Check if element is visible
  isVisible: (element) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    return element && element.offsetParent !== null;
  }
};
window.RenderUtils ??= {};

RenderUtils.list = (container, items) => {
  if (!container) return;
  container.innerHTML = '';

  if (!items || items.length === 0) {
    container.innerHTML =
      '<p style="color:#64748b">No recommendations available.</p>';
    return;
  }

  const ul = document.createElement('ul');
  items.forEach(txt => {
    const li = document.createElement('li');
    li.textContent = txt;
    ul.appendChild(li);
  });
  container.appendChild(ul);
};
// String Utilities
const StringUtils = {
  // Capitalize first letter
  capitalize: (str) => str.charAt(0).toUpperCase() + str.slice(1),
  
  // Convert to kebab-case
  kebabCase: (str) => str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase(),
  
  // Convert to camelCase
  camelCase: (str) => str.replace(/-([a-z])/g, (g) => g[1].toUpperCase()),
  
  // Truncate string
  truncate: (str, length, suffix = '...') => {
    if (str.length <= length) return str;
    return str.substring(0, length) + suffix;
  },
  
  // Remove HTML tags
  stripHtml: (str) => str.replace(/<[^>]*>/g, ''),
  
  // Escape HTML
  escapeHtml: (str) => {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
  
  // Generate random string
  random: (length = 8) => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  },
  
  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
};

// Array Utilities
const ArrayUtils = {
  // Remove duplicates
  unique: (arr) => [...new Set(arr)],
  
  // Chunk array
  chunk: (arr, size) => {
    const chunks = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  },
  
  // Shuffle array
  shuffle: (arr) => {
    const shuffled = [...arr];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  },
  
  // Get random item
  random: (arr) => arr[Math.floor(Math.random() * arr.length)],
  
  // Group by property
  groupBy: (arr, key) => {
    return arr.reduce((groups, item) => {
      const group = item[key];
      groups[group] = groups[group] || [];
      groups[group].push(item);
      return groups;
    }, {});
  }
};

// Object Utilities
const ObjectUtils = {
  // Deep clone
  deepClone: (obj) => JSON.parse(JSON.stringify(obj)),
  
  // Merge objects
  merge: (target, ...sources) => {
    return Object.assign({}, target, ...sources);
  },
  
  // Get nested property
  get: (obj, path, defaultValue = null) => {
    const keys = path.split('.');
    let result = obj;
    
    for (const key of keys) {
      if (result === null || result === undefined) {
        return defaultValue;
      }
      result = result[key];
    }
    
    return result !== undefined ? result : defaultValue;
  },
  
  // Set nested property
  set: (obj, path, value) => {
    const keys = path.split('.');
    let current = obj;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }
    
    current[keys[keys.length - 1]] = value;
    return obj;
  },
  
  // Check if object is empty
  isEmpty: (obj) => Object.keys(obj).length === 0,
  
  // Pick properties
  pick: (obj, keys) => {
    const result = {};
    keys.forEach(key => {
      if (key in obj) {
        result[key] = obj[key];
      }
    });
    return result;
  },
  
  // Omit properties
  omit: (obj, keys) => {
    const result = { ...obj };
    keys.forEach(key => {
      delete result[key];
    });
    return result;
  }
};

// Date Utilities
const DateUtils = {
  // Format date
  format: (date, format = 'YYYY-MM-DD') => {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    
    return format
      .replace('YYYY', year)
      .replace('MM', month)
      .replace('DD', day)
      .replace('HH', hours)
      .replace('mm', minutes)
      .replace('ss', seconds);
  },
  
  // Get relative time
  relative: (date) => {
    const now = new Date();
    const diff = now - new Date(date);
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    return 'Just now';
  },
  
  // Check if date is today
  isToday: (date) => {
    const today = new Date();
    const d = new Date(date);
    return d.toDateString() === today.toDateString();
  },
  
  // Add days to date
  addDays: (date, days) => {
    const result = new Date(date);
    result.setDate(result.getDate() + days);
    return result;
  }
};

// URL Utilities
const UrlUtils = {
  // Get query parameters
  getParams: () => {
    const params = {};
    const searchParams = new URLSearchParams(window.location.search);
    for (const [key, value] of searchParams) {
      params[key] = value;
    }
    return params;
  },
  
  // Get single parameter
  getParam: (name, defaultValue = null) => {
    const params = new URLSearchParams(window.location.search);
    return params.get(name) || defaultValue;
  },
  
  // Set parameter
  setParam: (name, value) => {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.replaceState({}, '', url);
  },
  
  // Remove parameter
  removeParam: (name) => {
    const url = new URL(window.location);
    url.searchParams.delete(name);
    window.history.replaceState({}, '', url);
  },
  
  // Build URL with parameters
  build: (base, params = {}) => {
    const url = new URL(base);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.set(key, value);
      }
    });
    return url.toString();
  }
};

// Validation Utilities
const ValidationUtils = {
  // Email validation
  isEmail: (email) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  },
  
  // URL validation
  isUrl: (url) => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  },
  
  // Phone validation (basic)
  isPhone: (phone) => {
    const regex = /^[\+]?[1-9][\d]{0,15}$/;
    return regex.test(phone.replace(/[\s\-\(\)]/g, ''));
  },
  
  // Required field
  required: (value) => {
    return value !== null && value !== undefined && value !== '';
  },
  
  // Minimum length
  minLength: (value, min) => {
    return value && value.length >= min;
  },
  
  // Maximum length
  maxLength: (value, max) => {
    return !value || value.length <= max;
  },
  
  // Numeric validation
  isNumeric: (value) => {
    return !isNaN(value) && !isNaN(parseFloat(value));
  },
  
  // File type validation
  isFileType: (file, types) => {
    const allowedTypes = Array.isArray(types) ? types : [types];
    return allowedTypes.includes(file.type);
  },
  
  // File size validation (in MB)
  isFileSize: (file, maxSizeMB) => {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
  }
};

// Animation Utilities
const AnimationUtils = {
  // Fade in element
  fadeIn: (element, duration = 300) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    
    element.style.opacity = '0';
    element.style.display = 'block';
    
    const start = performance.now();
    
    const fade = (timestamp) => {
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      
      element.style.opacity = progress;
      
      if (progress < 1) {
        requestAnimationFrame(fade);
      }
    };
    
    requestAnimationFrame(fade);
  },
  
  // Fade out element
  fadeOut: (element, duration = 300) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    
    const start = performance.now();
    const startOpacity = parseFloat(getComputedStyle(element).opacity);
    
    const fade = (timestamp) => {
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      
      element.style.opacity = startOpacity * (1 - progress);
      
      if (progress < 1) {
        requestAnimationFrame(fade);
      } else {
        element.style.display = 'none';
      }
    };
    
    requestAnimationFrame(fade);
  },
  
  // Slide down
  slideDown: (element, duration = 300) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    
    element.style.height = '0';
    element.style.overflow = 'hidden';
    element.style.display = 'block';
    
    const targetHeight = element.scrollHeight;
    const start = performance.now();
    
    const slide = (timestamp) => {
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      
      element.style.height = (targetHeight * progress) + 'px';
      
      if (progress < 1) {
        requestAnimationFrame(slide);
      } else {
        element.style.height = '';
        element.style.overflow = '';
      }
    };
    
    requestAnimationFrame(slide);
  },
  
  // Slide up
  slideUp: (element, duration = 300) => {
    if (typeof element === 'string') {
      element = document.querySelector(element);
    }
    
    const startHeight = element.offsetHeight;
    const start = performance.now();
    
    element.style.overflow = 'hidden';
    
    const slide = (timestamp) => {
      const elapsed = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      
      element.style.height = (startHeight * (1 - progress)) + 'px';
      
      if (progress < 1) {
        requestAnimationFrame(slide);
      } else {
        element.style.display = 'none';
        element.style.height = '';
        element.style.overflow = '';
      }
    };
    
    requestAnimationFrame(slide);
  }
};

// Debounce and Throttle
const FunctionUtils = {
  // Debounce function
  debounce: (func, delay) => {
    let timeoutId;
    return function(...args) {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
  },
  
  // Throttle function
  throttle: (func, limit) => {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },
  
  // Once function (executes only once)
  once: (func) => {
    let called = false;
    return function(...args) {
      if (!called) {
        called = true;
        return func.apply(this, args);
      }
    };
  }
};

// Export utilities to global scope
window.DOM = DOM;
window.StringUtils = StringUtils;
window.ArrayUtils = ArrayUtils;
window.ObjectUtils = ObjectUtils;
window.DateUtils = DateUtils;
window.UrlUtils = UrlUtils;
window.ValidationUtils = ValidationUtils;
window.AnimationUtils = AnimationUtils;
window.FunctionUtils = FunctionUtils;