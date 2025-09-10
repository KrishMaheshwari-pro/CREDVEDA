"""
This file contains the HTML, CSS, and JavaScript assets for the
self-contained, rule-based chatbot.
"""

def get_chatbot_assets():
    """
    Returns the HTML, CSS, and JavaScript for the chatbot as a dictionary.
    This allows the main Flask app to serve these assets dynamically.
    """
    chatbot_html = """
    <!-- Chatbot container that will be injected into the page -->
    <div id="chatbot-container" class="hidden">
        <div id="chatbot-header">
            <span>CredVeda Assistant</span>
            <button id="close-chatbot-btn" aria-label="Close Chatbot">&times;</button>
        </div>
        <div id="chatbot-messages">
            <!-- Messages will be dynamically added here -->
        </div>
        <div id="chatbot-suggestions">
            <!-- Suggested questions will be dynamically added here -->
        </div>
        <div id="chatbot-input-container">
            <input type="text" id="chatbot-input" placeholder="Ask a question..." aria-label="Chatbot Input">
            <button id="chatbot-send-btn" aria-label="Send Message">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>
    <!-- Button to open the chatbot -->
    <button id="open-chatbot-btn" aria-label="Open Chatbot">
        <i class="fas fa-comment-dots"></i>
    </button>
    """

    chatbot_css = """
    /* Floating button to open the chatbot */
    #open-chatbot-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #2563eb;
        color: white;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        border: none;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease-in-out, background-color 0.2s;
    }
    #open-chatbot-btn:hover {
        transform: scale(1.1);
        background-color: #1d4ed8;
    }

    /* Main chatbot window */
    #chatbot-container {
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 360px;
        max-width: 90vw;
        height: 520px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        transition: opacity 0.3s, transform 0.3s;
        transform-origin: bottom right;
    }
    #chatbot-container.hidden {
        opacity: 0;
        transform: scale(0.5);
        pointer-events: none;
    }

    /* Chatbot header section */
    #chatbot-header {
        background-color: #21262d;
        color: #e6edf3;
        padding: 15px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #30363d;
    }
    #close-chatbot-btn {
        background: none;
        border: none;
        color: #e6edf3;
        font-size: 24px;
        cursor: pointer;
    }

    /* Message display area */
    #chatbot-messages {
        flex-grow: 1;
        padding: 15px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .chatbot-message {
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 85%;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #2563eb;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
    }
    .bot-message {
        background-color: #30363d;
        color: #e6edf3;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
    }
    .bot-message a {
        color: #58a6ff;
        text-decoration: underline;
    }
    .bot-message a:hover {
        color: #79c0ff;
    }

    /* Suggested Questions */
    #chatbot-suggestions {
        padding: 10px 15px 5px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        border-top: 1px solid #30363d;
    }
    .suggestion-btn {
        background-color: #30363d;
        color: #e6edf3;
        border: 1px solid #444c56;
        border-radius: 15px;
        padding: 8px 12px;
        font-size: 13px;
        cursor: pointer;
        transition: background-color 0.2s, border-color 0.2s;
    }
    .suggestion-btn:hover {
        background-color: #444c56;
        border-color: #58a6ff;
    }

    /* Input area */
    #chatbot-input-container {
        display: flex;
        padding: 15px;
        border-top: 1px solid #30363d;
        background-color: #21262d;
    }
    #chatbot-input {
        flex-grow: 1;
        border: 1px solid #30363d;
        background-color: #0d1117;
        color: #e6edf3;
        border-radius: 6px;
        padding: 10px;
        margin-right: 10px;
    }
    #chatbot-send-btn {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0 15px;
        cursor: pointer;
        font-size: 16px;
    }

    /* Light Theme Adjustments */
    body.light-theme #chatbot-container { background-color: #ffffff; border-color: #e2e8f0; }
    body.light-theme #chatbot-header { background-color: #f0f2f5; color: #1a202c; border-color: #e2e8f0; }
    body.light-theme #close-chatbot-btn { color: #1a202c; }
    body.light-theme #chatbot-messages { background-color: #f9fafb; }
    body.light-theme .bot-message { background-color: #e2e8f0; color: #1a202c; }
    body.light-theme #chatbot-input-container { background-color: #f0f2f5; border-color: #e2e8f0; }
    body.light-theme #chatbot-input { background-color: #ffffff; color: #1a202c; border-color: #cbd5e0; }
    body.light-theme #chatbot-suggestions { border-color: #e2e8f0; }
    body.light-theme .suggestion-btn { background-color: #e2e8f0; color: #1a202c; border-color: #cbd5e0; }
    body.light-theme .suggestion-btn:hover { background-color: #cbd5e0; border-color: #007bff; }
    """

    chatbot_js = """
    document.addEventListener('DOMContentLoaded', () => {
        const openBtn = document.getElementById('open-chatbot-btn');
        const closeBtn = document.getElementById('close-chatbot-btn');
        const container = document.getElementById('chatbot-container');
        const sendBtn = document.getElementById('chatbot-send-btn');
        const input = document.getElementById('chatbot-input');
        const messagesContainer = document.getElementById('chatbot-messages');
        const suggestionsContainer = document.getElementById('chatbot-suggestions');

        if (!openBtn || !closeBtn || !container || !sendBtn || !input || !messagesContainer || !suggestionsContainer) {
            console.error('Chatbot elements not found!');
            return;
        }

        const chatbotData = {
            'default': "I'm sorry, I didn't quite understand that. You can ask me about CredVeda's features, how credit scoring works, or how to contact support.",
            'greeting': "Hello! I'm the CredVeda assistant. How can I help you today? Here are a few things you can ask:",
            'features': "CredVeda uses AI for predictive credit scoring (XGBoost), provides transparent explanations (XAI with SHAP), analyzes news sentiment (NLP), and sends real-time anomaly alerts. You can learn more on the <a href='/ai-features'>AI Features</a> page.",
            'scoring': "Our credit scoring is powered by an XGBoost machine learning model. It analyzes financial data, market trends, and news sentiment to generate a score. We use SHAP to explain how each factor contributes to the score.",
            'dashboard': "The dashboard provides a real-time overview of a company's credit score, market volatility, and macroeconomic indicators. You can explore score trends over time and see a detailed breakdown of what influences the score.",
            'contact': "You can reach our support team by visiting the <a href='/contact'>Contact Us</a> page. We're happy to help!",
            'bye': "Goodbye! Feel free to ask if you have more questions."
        };

        const keywords = {
            'features': ['feature', 'capability', 'what can you do', "what are credveda's features?"],
            'scoring': ['score', 'scoring', 'how it works', 'xgboost', 'shap', 'model', 'how does credit scoring work?'],
            'dashboard': ['dashboard', 'chart', 'graph', 'insight'],
            'contact': ['contact', 'support', 'help', 'email', 'phone', 'how can i contact support?'],
            'greeting': ['hello', 'hi', 'hey', 'yo'],
            'bye': ['bye', 'goodbye', 'see you', 'later']
        };

        const suggestedQuestions = [
            "What are CredVeda's features?",
            "How does credit scoring work?",
            "How can I contact support?"
        ];

        function renderSuggestions() {
            suggestionsContainer.innerHTML = '';
            suggestedQuestions.forEach(q => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = q;
                btn.addEventListener('click', () => {
                    input.value = q;
                    handleUserInput();
                });
                suggestionsContainer.appendChild(btn);
            });
            suggestionsContainer.style.display = 'flex';
        }

        function hideSuggestions() {
            suggestionsContainer.style.display = 'none';
        }

        function getBotResponse(userInput) {
            const lowerInput = userInput.toLowerCase();
            for (const key in keywords) {
                for (const keyword of keywords[key]) {
                    if (lowerInput.includes(keyword)) {
                        return chatbotData[key];
                    }
                }
            }
            return chatbotData['default'];
        }

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chatbot-message ${sender}-message`;
            messageDiv.innerHTML = text; // Use innerHTML to render potential links
            messagesContainer.appendChild(messageDiv);
            // Scroll to the bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function handleUserInput() {
            const userInput = input.value.trim();
            if (userInput) {
                addMessage(userInput, 'user');
                const botResponse = getBotResponse(userInput);
                // Simulate bot thinking
                setTimeout(() => addMessage(botResponse, 'bot'), 500);
                input.value = '';
                hideSuggestions(); // Hide suggestions after first interaction
            }
        }

        openBtn.addEventListener('click', () => {
            container.classList.remove('hidden');
            // Show suggestions only if the conversation has just started
            if (messagesContainer.children.length <= 1) {
                renderSuggestions();
            }
        });
        closeBtn.addEventListener('click', () => container.classList.add('hidden'));
        sendBtn.addEventListener('click', handleUserInput);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleUserInput();
            }
        });
        
        // Add an initial greeting message after a short delay
        setTimeout(() => {
            if (messagesContainer.children.length === 0) {
                 addMessage(chatbotData['greeting'], 'bot');
                 renderSuggestions();
            }
        }, 1500);
    });
    """
    return {
        "html": chatbot_html,
        "css": chatbot_css,
        "js": chatbot_js
    }
