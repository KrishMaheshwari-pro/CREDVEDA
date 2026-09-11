"""
This file contains the HTML, CSS, and JavaScript assets for the
interactive Conversational Underwriter.
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
            <span>CredVeda AI Underwriter</span>
            <button id="close-chatbot-btn" aria-label="Close Chatbot">&times;</button>
        </div>
        <div id="chatbot-messages">
            <!-- Messages will be dynamically added here -->
        </div>
        <div id="chatbot-suggestions">
            <!-- Suggested questions will be dynamically added here -->
        </div>
        <div id="chatbot-input-container">
            <input type="text" id="chatbot-input" placeholder="Type data signals (e.g. 'monthly revenue is 2 lakhs')..." aria-label="Chatbot Input">
            <button id="chatbot-send-btn" aria-label="Send Message">
                <i class="fas fa-paper-plane"></i>
            </button>
        </div>
    </div>
    <!-- Button to open the chatbot -->
    <button id="open-chatbot-btn" aria-label="Open Chatbot">
        <i class="fas fa-robot"></i>
    </button>
    """

    chatbot_css = """
    #open-chatbot-btn {
        position: fixed; bottom: 26px; right: 26px;
        width: 54px; height: 54px; border-radius: 50%;
        background: var(--brand); color: #fff; border: none;
        font-size: 20px; cursor: pointer; z-index: 999;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 8px 24px rgba(0,0,0,.3);
        transition: transform .18s ease, background .18s;
    }
    #open-chatbot-btn:hover { transform: scale(1.07); background: var(--brand-hi); }

    #chatbot-container {
        position: fixed; bottom: 92px; right: 26px;
        width: 360px; max-width: calc(100vw - 32px); height: 520px; max-height: calc(100vh - 140px);
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; box-shadow: var(--shadow-lg);
        z-index: 1000; display: flex; flex-direction: column; overflow: hidden;
        transition: opacity .22s ease, transform .22s ease; transform-origin: bottom right;
    }
    #chatbot-container.hidden { opacity: 0; transform: scale(.92) translateY(8px); pointer-events: none; }

    #chatbot-header {
        background: var(--surface-2); color: var(--text);
        padding: 14px 16px; font-weight: 650; font-size: .9375rem;
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--border);
    }
    #close-chatbot-btn { background: none; border: none; color: var(--text-dim); font-size: 22px; cursor: pointer; line-height: 1; }
    #close-chatbot-btn:hover { color: var(--text); }

    #chatbot-messages {
        flex-grow: 1; padding: 16px; overflow-y: auto;
        display: flex; flex-direction: column; gap: 10px;
    }
    .chatbot-message {
        padding: 10px 14px; border-radius: 14px; max-width: 88%;
        line-height: 1.55; font-size: .875rem; word-wrap: break-word;
    }
    .user-message { background: var(--brand); color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; }
    .bot-message  { background: var(--surface-3); color: var(--text); align-self: flex-start; border-bottom-left-radius: 4px; }
    .bot-message a { color: var(--brand); text-decoration: underline; }

    #chatbot-suggestions {
        padding: 10px 14px 4px; display: flex; flex-wrap: wrap; gap: 7px;
        border-top: 1px solid var(--border-soft);
    }
    .suggestion-btn {
        background: var(--surface-2); color: var(--text-dim);
        border: 1px solid var(--border); border-radius: 999px;
        padding: 7px 12px; font-size: .75rem; font-family: inherit; cursor: pointer;
        transition: all .15s;
    }
    .suggestion-btn:hover { border-color: var(--brand-line); color: var(--brand); }

    #chatbot-input-container {
        display: flex; gap: 8px; padding: 13px 14px;
        border-top: 1px solid var(--border-soft); background: var(--surface-2);
    }
    #chatbot-input {
        flex-grow: 1; border: 1px solid var(--border); background: var(--bg-soft);
        color: var(--text); border-radius: 9px; padding: 9px 12px;
        font-family: inherit; font-size: .875rem;
    }
    #chatbot-input:focus { outline: none; border-color: var(--brand); }
    #chatbot-send-btn {
        background: var(--brand); color: #fff; border: none; border-radius: 9px;
        padding: 0 15px; cursor: pointer; font-size: 15px;
    }
    #chatbot-send-btn:hover { background: var(--brand-hi); }
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

        const suggestedQuestions = [
            "Revenue is 2 lakhs/mo and 30% digital. Score?",
            "What if they also have a 650 bureau score?",
            "Does paying GST regularly improve the score?"
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

        let contextScore = 450; // Starting baseline

        function getBotResponse(userInput) {
            const lower = userInput.toLowerCase();
            if (lower.includes("revenue") || lower.includes("lakhs") || lower.includes("digital")) {
                contextScore += 180;
                return `Got it! Injecting 2L/mo revenue with 30% digital footprint.<br><br><b>Provisional Score:</b> <span style="color:var(--good); font-weight:bold">${contextScore}</span><br><br><b>Explanation:</b> Demonstrable cash flow directly reduces default probability. The digital fraction makes it verifiable.`;
            } else if (lower.includes("bureau") || lower.includes("650")) {
                contextScore += 50;
                return `Factoring in the 650 bureau score...<br><br><b>Provisional Score:</b> <span style="color:var(--good); font-weight:bold">${contextScore}</span><br><br><b>Explanation:</b> A thin but positive bureau file provides a small baseline boost to the alternative signals.`;
            } else if (lower.includes("gst")) {
                contextScore += 75;
                return `Adding consistent GST history...<br><br><b>Provisional Score:</b> <span style="color:var(--good); font-weight:bold">${contextScore}</span><br><br><b>Explanation:</b> High GST regularity indicates formalized business practices and lowers risk significantly.`;
            } else {
                return "I'm listening. Tell me about the applicant's cash flow, digital adoption, GST regularity, or existing loan burdens to see how the score reacts.";
            }
        }

        function addMessage(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chatbot-message ${sender}-message`;
            messageDiv.innerHTML = text; // Use innerHTML to render potential links
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function handleUserInput() {
            const userInput = input.value.trim();
            if (userInput) {
                addMessage(userInput, 'user');
                const botResponse = getBotResponse(userInput);
                setTimeout(() => addMessage(botResponse, 'bot'), 500);
                input.value = '';
                hideSuggestions(); 
            }
        }

        openBtn.addEventListener('click', () => {
            container.classList.remove('hidden');
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
        
        setTimeout(() => {
            if (messagesContainer.children.length === 0) {
                 addMessage("Hello! I'm the AI Underwriting Assistant. You can dynamically test how alternative data signals affect an applicant's score here.", 'bot');
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
