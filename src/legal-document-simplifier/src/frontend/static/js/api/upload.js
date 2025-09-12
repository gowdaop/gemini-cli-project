// API Classes
class UploadAPI {
    constructor() {
        this.baseURL = 'http://localhost:8000'; // Update for production
        this.apiKey = 'legal-doc-analyzer-2025-secure-key-f47d4a2c';
    }

    async uploadDocument(file, progressCallback) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Use XMLHttpRequest for progress tracking
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();

                // Upload progress
                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable && progressCallback) {
                        const percentComplete = (event.loaded / event.total) * 100;
                        progressCallback(percentComplete);
                    }
                };

                // Success handler
                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid response format'));
                        }
                    } else {
                        try {
                            const errorData = JSON.parse(xhr.responseText);
                            reject(new Error(errorData?.detail || `Upload failed with status ${xhr.status}`));
                        } catch (e) {
                            reject(new Error(`Upload failed with status ${xhr.status}`));
                        }
                    }
                };

                // Error handler
                xhr.onerror = () => {
                    reject(new Error('Network error during upload'));
                };

                // Configure request
                xhr.open('POST', `${this.baseURL}/upload/`);
                xhr.setRequestHeader('x-api-key', this.apiKey);
                xhr.timeout = 300000; // 5 minute timeout

                // Send request
                xhr.send(formData);
            });
        } catch (error) {
            console.error('Upload API error:', error);
            throw error;
        }
    }
}

class AnalysisAPI {
    constructor() {
        this.baseURL = 'http://localhost:8000';
        this.apiKey = 'legal-doc-analyzer-2025-secure-key-f47d4a2c';
    }

    async analyzeDocument(data, progressCallback) {
        try {
            const response = await fetch(`${this.baseURL}/analyze/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.apiKey
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `Analysis failed with status ${response.status}`);
            }

            const result = await response.json();
            
            // Simulate progress callback
            if (progressCallback) {
                progressCallback(100);
            }

            return result;
        } catch (error) {
            console.error('Analysis API error:', error);
            throw error;
        }
    }
}
