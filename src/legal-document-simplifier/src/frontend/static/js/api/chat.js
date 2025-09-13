// Chat API Client - Fixed for Browser Compatibility
class ChatAPI {
    constructor() {
        this.baseURL = 'http://localhost:8000';
        this.apiKey = 'legal-doc-analyzer-2025-secure-key-f47d4a2c';
        this.timeout = 25000; // 25 seconds
    }

    // ✅ FIXED: Custom timeout implementation for better browser support
    createTimeoutPromise(promise, timeoutMs) {
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Request timed out. Please try again.'));
            }, timeoutMs);

            promise
                .then((response) => {
                    clearTimeout(timeoutId);
                    resolve(response);
                })
                .catch((error) => {
                    clearTimeout(timeoutId);
                    reject(error);
                });
        });
    }

    async sendMessage(message, conversationId = null, ocr = null, summaryHint = null) {
        try {
            const requestBody = {
                question: message,
                conversation_id: conversationId,
                ...(ocr && { ocr }),
                ...(summaryHint && { summary_hint: summaryHint })
            };

            const fetchPromise = fetch(`${this.baseURL}/chat/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': this.apiKey
                },
                body: JSON.stringify(requestBody)
            });

            // ✅ Use custom timeout instead of AbortSignal.timeout
            const response = await this.createTimeoutPromise(fetchPromise, this.timeout);

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `Request failed with status ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Chat API sendMessage error:', error);
            throw error;
        }
    }

    async getConversationHistory(conversationId) {
        try {
            const fetchPromise = fetch(`${this.baseURL}/chat/conversations/${conversationId}`, {
                method: 'GET',
                headers: {
                    'x-api-key': this.apiKey
                }
            });

            const response = await this.createTimeoutPromise(fetchPromise, 10000);

            if (response.status === 404) {
                return null;
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `Request failed with status ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Chat API getConversationHistory error:', error);
            throw error;
        }
    }

    async deleteConversation(conversationId) {
        try {
            const fetchPromise = fetch(`${this.baseURL}/chat/conversations/${conversationId}`, {
                method: 'DELETE',
                headers: {
                    'x-api-key': this.apiKey
                }
            });

            const response = await this.createTimeoutPromise(fetchPromise, 10000);

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || `Request failed with status ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Chat API deleteConversation error:', error);
            throw error;
        }
    }
}

// ✅ FIXED: Proper module export for compatibility
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChatAPI };
} else if (typeof window !== 'undefined') {
    window.ChatAPI = ChatAPI;
}
