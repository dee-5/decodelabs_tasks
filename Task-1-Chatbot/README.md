# Project 1: Rule-Based AI Chatbot

**DecodeLabs Industrial Training Kit — Batch 2026**

A simple, rule-based chatbot built in Python that responds to predefined user
inputs using dictionary lookups and control-flow logic. This is the
foundational project before moving into machine-learning-based chatbots —
it focuses on **control flow, decision-making logic, and basic AI concepts**.

## 🧠 How It Works

1. **Input Loop** – the bot runs inside a `while True` loop and keeps
   listening until the user types an exit command.
2. **Sanitization** – every message is lowercased and stripped of extra
   spaces, so `Hello`, `hello `, and `HELLO` all match the same rule.
3. **Knowledge Base** – a Python dictionary stores `{"trigger": "reply"}`
   pairs (10 intents included, more than the required 5).
4. **Fallback** – if the input doesn't match any rule, the bot returns a
   random "I don't understand" style response instead of crashing.
5. **Exit Strategy** – typing `bye`, `exit`, `quit`, `goodbye`, or
   `see you` cleanly breaks the loop and ends the chat.

## 🚀 How to Run

```bash
python chatbot.py
```

Then just start typing! Example:
RuleBot: Good evening! I'm RuleBot 🤖. Type 'bye' anytime to exit.
You: hello
RuleBot: Hi there! How can I help you today?
You: what is your name
RuleBot: I'm RuleBot, a simple rule-based chatbot built for Project 1.
You: bye
RuleBot: Goodbye! Have a great day. 👋

## 🛠️ Skills Demonstrated

- Control flow (`if`/`elif`, loops)
- Dictionaries / hash maps for O(1) lookups instead of long if-elif ladders
- Input sanitization and normalization
- Basic conversational state handling
- Graceful fallback and exit handling

## 📌 Notes / Possible Extensions

- Add more intents to the `responses` dictionary
- Add nested conditions for multi-step conversations (e.g. asking follow-up
  questions)
- Give the bot a more distinct personality
- Later projects will replace this exact-match dictionary with semantic
  (vector-based) matching using embeddings

---
Built as part of the DecodeLabs AI Engineering Internship, 2026.
