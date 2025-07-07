// Georgian Chatbot Frontend JavaScript
const form = document.getElementById('chatbot-form');
const questionInput = document.getElementById('question');
const submitBtn = document.getElementById('submit-btn');
const answerSection = document.getElementById('answer');

// Chat history array
let chatHistory = [];

// Initialize chat interface
function initializeChat() {
    // Add welcome message
    displayMessage('მოგესალმებით! მე ვარ ქართული ჩატბოტი. რითი შემიძლია დაგეხმაროთ?', 'bot');

    // Focus on input
    questionInput.focus();

    // Load chat history from localStorage if available
    loadChatHistory();
}

// Handle form submission
form.addEventListener('submit', function (e) {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question) {
        showError('გთხოვთ, შეიყვანეთ კითხვა');
        return;
    }

    // Add user message to chat
    displayMessage(question, 'user');

    // Clear input and disable submit button
    questionInput.value = '';
    setLoading(true);

    // Send question to server
    sendQuestion(question);
});

// Send question to server
async function sendQuestion(question) {
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (data.success) {
            displayMessage(data.answer, 'bot');

            // Add to chat history
            chatHistory.push({
                question: question,
                answer: data.answer,
                timestamp: new Date().toISOString()
            });

            // Save to localStorage
            saveChatHistory();
        } else {
            showError(data.error || 'შეცდომა მოხდა');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('კავშირის შეცდომა. გთხოვთ, მოგვიანებით სცადოთ.');
    } finally {
        setLoading(false);
    }
}

// Display message in chat
function displayMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = text;

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString('ka-GE');

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);

    answerSection.appendChild(messageDiv);

    // Scroll to bottom
    answerSection.scrollTop = answerSection.scrollHeight;
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;

    answerSection.appendChild(errorDiv);

    // Remove error after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);

    // Scroll to bottom
    answerSection.scrollTop = answerSection.scrollHeight;
}

// Set loading state
function setLoading(isLoading) {
    if (isLoading) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'იტვირთება...';
        questionInput.disabled = true;

        // Add typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
        answerSection.appendChild(typingDiv);
    } else {
        submitBtn.disabled = false;
        submitBtn.textContent = 'გაგზავნა';
        questionInput.disabled = false;
        questionInput.focus();

        // Remove typing indicator
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Save chat history to localStorage
function saveChatHistory() {
    try {
        localStorage.setItem('georgian_chat_history', JSON.stringify(chatHistory));
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}

// Load chat history from localStorage
function loadChatHistory() {
    try {
        const saved = localStorage.getItem('georgian_chat_history');
        if (saved) {
            chatHistory = JSON.parse(saved);

            // Display recent messages (last 10)
            const recentHistory = chatHistory.slice(-10);
            recentHistory.forEach(item => {
                displayMessage(item.question, 'user');
                displayMessage(item.answer, 'bot');
            });
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        chatHistory = [];
    }
}

// Clear chat history
function clearChatHistory() {
    chatHistory = [];
    localStorage.removeItem('georgian_chat_history');
    answerSection.innerHTML = '';
    displayMessage('მოგესალმებით! მე ვარ ქართული ჩატბოტი. რითი შემიძლია დაგეხმაროთ?', 'bot');
}

// Add clear button functionality
const clearBtn = document.getElementById('clear-chat');
if (clearBtn) {
    clearBtn.addEventListener('click', clearChatHistory);
}

// Handle enter key in input
questionInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        form.dispatchEvent(new Event('submit'));
    }
});

// Initialize the chat interface
initializeChat();