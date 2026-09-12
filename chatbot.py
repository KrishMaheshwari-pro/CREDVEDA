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
            <input type="text" id="chatbot-input" placeholder="Describe an applicant, or ask why a score is what it is…" aria-label="Chatbot Input">
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

    .bot-message b { color: var(--text); }
    .bot-message ul { margin: .4rem 0 0; padding-left: 1.1rem; }

    /* three-dot "thinking" pulse while the model scores */
    .typing { display: inline-flex; gap: 4px; align-items: center; height: 1.1em; }
    .typing i {
        width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim);
        display: inline-block; animation: cv-blink 1.25s infinite ease-in-out;
    }
    .typing i:nth-child(2) { animation-delay: .18s; }
    .typing i:nth-child(3) { animation-delay: .36s; }
    @keyframes cv-blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }

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

        // The applicant whose page we are on, if any. The server answers
        // "why this score" / "how do they improve" from live data for them.
        const pathMatch = window.location.pathname.match(
            /\\/(?:applicant|passport|score-report|improvement|fundings)\\/([A-Za-z0-9_-]+)/);
        const APPLICANT_ID = pathMatch ? pathMatch[1] : '';

        // Conversation state lives here, not on the server, so the backend
        // stays stateless across workers. `profile` accumulates the signals
        // mentioned so far; `lastResult` lets the server report a delta.
        let profile = {};
        let lastResult = null;
        let busy = false;

        const GENERIC = [
            "Kirana shop, revenue 2.5 lakhs/mo, 40% digital, GST filed",
            "What if their bureau score was 720?",
            "What data do you actually need?"
        ];
        const ON_APPLICANT = [
            "Why this score?",
            "How can they improve it?",
            "Can they afford the loan?",
            "Which schemes do they qualify for?"
        ];
        const FOLLOW_UP = [
            "What if digital adoption were 80%?",
            "What if they had no existing EMI?",
            "How do they improve?",
            "Which schemes now?"
        ];

        function renderSuggestions(list) {
            suggestionsContainer.innerHTML = '';
            list.forEach(q => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = q;
                btn.addEventListener('click', () => { input.value = q; handleUserInput(); });
                suggestionsContainer.appendChild(btn);
            });
            suggestionsContainer.style.display = list.length ? 'flex' : 'none';
        }

        function hideSuggestions() { suggestionsContainer.style.display = 'none'; }

        function addMessage(html, sender) {
            const div = document.createElement('div');
            div.className = `chatbot-message ${sender}-message`;
            div.innerHTML = html;
            messagesContainer.appendChild(div);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return div;
        }

        function showTyping() {
            const div = addMessage(
                '<span class="typing"><i></i><i></i><i></i></span>', 'bot');
            div.id = 'typing-indicator';
            return div;
        }

        // User text goes in as textContent, never as HTML -- the bot's own
        // replies are trusted markup, but whatever is typed into the box is not.
        function escapeHtml(s) {
            const d = document.createElement('div');
            d.textContent = s;
            return d.innerHTML;
        }

        async function handleUserInput() {
            const userInput = input.value.trim();
            if (!userInput || busy) return;
            busy = true;
            addMessage(escapeHtml(userInput), 'user');
            input.value = '';
            hideSuggestions();
            const typing = showTyping();

            try {
                const res = await fetch('/api/chatbot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: userInput,
                        applicant_id: APPLICANT_ID,
                        profile: profile,
                        last_result: lastResult
                    })
                });
                const data = await res.json();
                typing.remove();
                if (!res.ok) {
                    addMessage(data.error || 'Something went wrong. Try again.', 'bot');
                } else {
                    addMessage(data.reply, 'bot');
                    if (data.profile) profile = data.profile;
                    lastResult = data.last_result || lastResult;
                    const hasProfile = profile && Object.keys(profile).length > 0;
                    renderSuggestions(hasProfile ? FOLLOW_UP
                                    : (APPLICANT_ID ? ON_APPLICANT : GENERIC));
                }
            } catch (e) {
                typing.remove();
                addMessage('I could not reach the scoring service. Check the connection and try again.', 'bot');
            } finally {
                busy = false;
                input.focus();
            }
        }

        function greet() {
            if (messagesContainer.children.length > 0) return;
            addMessage(
                APPLICANT_ID
                  ? `Hello. I'm the CredVeda underwriting assistant, running the real scoring model. ` +
                    `I can see <b>${escapeHtml(APPLICANT_ID)}</b> on screen — ask me why they scored ` +
                    `what they did, or describe a different applicant and I'll score them.`
                  : `Hello. I'm the CredVeda underwriting assistant, and I run the real scoring model ` +
                    `— describe an applicant in plain words and I'll score them, then change one ` +
                    `detail and I'll tell you how much it moved.`,
                'bot');
            renderSuggestions(APPLICANT_ID ? ON_APPLICANT : GENERIC);
        }

        openBtn.addEventListener('click', () => {
            container.classList.remove('hidden');
            greet();
            input.focus();
        });
        closeBtn.addEventListener('click', () => container.classList.add('hidden'));
        sendBtn.addEventListener('click', handleUserInput);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleUserInput();
        });
    });
    """
    return {
        "html": chatbot_html,
        "css": chatbot_css,
        "js": chatbot_js
    }
