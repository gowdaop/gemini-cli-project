/**
 * API Client for LEGAL-SIMPLIFIER
 * Modern fetch-based HTTP client with authentication and error handling
 */

class APIClient {
    constructor(config = {}) {
        this.baseURL = config.baseURL || 'http://localhost:8000';
        this.apiKey = config.apiKey || 'legal-doc-analyzer-2025-secure-key-f47d4a2c';
        this.timeout = config.timeout || 30000;
        this.onError = config.onError || ((error) => console.error(error));
        this.onLoading = config.onLoading || (() => {});
        this.requestInterceptors = [];
        this.responseInterceptors = [];
    }

    // Core HTTP methods
    async get(endpoint, options = {}) {
        return this.request('GET', endpoint, null, options);
    }

    async post(endpoint, data = null, options = {}) {
        return this.request('POST', endpoint, data, options);
    }

    async put(endpoint, data = null, options = {}) {
        return this.request('PUT', endpoint, data, options);
    }

    async delete(endpoint, options = {}) {
        return this.request('DELETE', endpoint, null, options);
    }

    async patch(endpoint, data = null, options = {}) {
        return this.request('PATCH', endpoint, data, options);
    }

    // Main request method
    async request(method, endpoint, data = null, options = {}) {
        const url = this.buildURL(endpoint);
        const config = await this.buildRequestConfig(method, data, options);

        this.onLoading(true);

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);

            config.signal = controller.signal;

            // Apply request interceptors
            for (const interceptor of this.requestInterceptors) {
                await interceptor(config);
            }

            const response = await fetch(url, config);
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw await this.handleErrorResponse(response);
            }

            let result;
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                result = await response.json();
            } else if (contentType && contentType.includes('text/')) {
                result = await response.text();
            } else {
                result = await response.blob();
            }

            // Apply response interceptors
            for (const interceptor of this.responseInterceptors) {
                result = await interceptor(result, response);
            }

            return result;

        } catch (error) {
            this.handleError(error);
            throw error;
        } finally {
            this.onLoading(false);
        }
    }

    // Upload method with progress tracking
    async upload(endpoint, file, options = {}) {
        const url = this.buildURL(endpoint);
        const formData = new FormData();
        
        if (file instanceof File) {
            formData.append('file', file);
        } else if (Array.isArray(file)) {
            file.forEach((f, index) => {
                formData.append(`file${index}`, f);
            });
        }

        // Add additional form data
        if (options.data) {
            Object.keys(options.data).forEach(key => {
                formData.append(key, options.data[key]);
            });
        }

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Progress tracking
            if (options.onProgress) {
                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable) {
                        const percentComplete = (event.loaded / event.total) * 100;
                        options.onProgress(percentComplete, event.loaded, event.total);
                    }
                };
            }

            xhr.onload = () => {
                try {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const contentType = xhr.getResponseHeader('content-type');
                        let result;
                        
                        if (contentType && contentType.includes('application/json')) {
                            result = JSON.parse(xhr.responseText);
                        } else {
                            result = xhr.responseText;
                        }
                        
                        resolve(result);
                    } else {
                        const error = new APIError(
                            `Upload failed: ${xhr.statusText}`,
                            xhr.status,
                            xhr.responseText
                        );
                        this.handleError(error);
                        reject(error);
                    }
                } catch (parseError) {
                    this.handleError(parseError);
                    reject(parseError);
                }
            };

            xhr.onerror = () => {
                const error = new APIError('Upload failed', 0, 'Network error');
                this.handleError(error);
                reject(error);
            };

            xhr.ontimeout = () => {
                const error = new APIError('Upload timeout', 0, 'Request timeout');
                this.handleError(error);
                reject(error);
            };

            xhr.open('POST', url);
            xhr.setRequestHeader('x-api-key', this.apiKey);
            xhr.timeout = this.timeout;

            // Add custom headers
            if (options.headers) {
                Object.keys(options.headers).forEach(header => {
                    xhr.setRequestHeader(header, options.headers[header]);
                });
            }

            xhr.send(formData);
        });
    }

    // Server-Sent Events for streaming
    createEventSource(endpoint, options = {}) {
        const url = this.buildURL(endpoint);
        const eventSource = new EventSource(url);

        eventSource.onopen = () => {
            if (options.onOpen) options.onOpen();
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (options.onMessage) options.onMessage(data);
            } catch (error) {
                console.error('Failed to parse SSE data:', error);
            }
        };

        eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            if (options.onError) options.onError(error);
        };

        return eventSource;
    }

    // Helper methods
    buildURL(endpoint) {
        const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
        return `${this.baseURL}${cleanEndpoint}`;
    }

    async buildRequestConfig(method, data, options) {
        const config = {
            method,
            headers: {
                'x-api-key': this.apiKey,
                ...options.headers
            }
        };

        // Handle different data types
        if (data !== null) {
            if (data instanceof FormData) {
                config.body = data;
                // Don't set Content-Type for FormData, let browser set it
            } else if (typeof data === 'object') {
                config.headers['Content-Type'] = 'application/json';
                config.body = JSON.stringify(data);
            } else {
                config.body = data;
            }
        }

        return config;
    }

    async handleErrorResponse(response) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        let errorData = null;

        try {
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                errorData = await response.json();
                errorMessage = errorData.message || errorData.detail || errorMessage;
            } else {
                errorData = await response.text();
            }
        } catch (parseError) {
            console.warn('Failed to parse error response:', parseError);
        }

        return new APIError(errorMessage, response.status, errorData);
    }

    handleError(error) {
        console.error('API Error:', error);
        this.onError(error);
    }

    // Interceptor methods
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }

    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    // Convenience methods for common endpoints
    async uploadDocument(file, options = {}) {
        return this.upload('/upload', file, {
            ...options,
            onProgress: (percent, loaded, total) => {
                console.log(`Upload progress: ${percent.toFixed(1)}%`);
                if (options.onProgress) options.onProgress(percent, loaded, total);
            }
        });
    }

    async analyzeDocument(documentId, options = {}) {
        return this.post('/analyze', { document_id: documentId }, options);
    }

    async chatWithDocument(documentId, message, options = {}) {
        return this.post('/chat', {
            document_id: documentId,
            message: message
        }, options);
    }

    async getDocumentResults(documentId, options = {}) {
        return this.get(`/results/${documentId}`, options);
    }

    async healthCheck() {
        return this.get('/healthz');
    }
}

// Custom Error class
class APIError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIClient, APIError };
}

// Global availability
window.APIClient = APIClient;
window.APIError = APIError;
export async function postJSON(url, body, { signal } = {}) {
  try {
    const controller = signal ? { signal } : new AbortController();
    const res = await fetch(url, {
      method : 'POST',
      headers: { 'Content-Type':'application/json' },
      body   : JSON.stringify(body),
      ...controller,
    });

    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();

  } catch (err) {
    if (err.name === 'AbortError') {
      console.info('Request was cancelled by the user -- safe to ignore');
      return;                     // swallow or propagate as you prefer
    }
    throw err;                    // real network/server error
  }
}
