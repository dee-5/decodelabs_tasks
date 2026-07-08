"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Batch 2026

Goal: A simple chatbot that responds to predefined user inputs
using if-else / dictionary logic, running in a continuous loop.
"""

import random
from datetime import datetime

# -------------------------------------------------
# PHASE 1: KNOWLEDGE BASE (the "brain" of the bot)
# -------------------------------------------------
# Each key is a cleaned-up user input, each value is the bot's reply.
# Using a dictionary instead of if-elif ladder = O(1) instant lookup.

responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! Nice to see you.",
    "hey": "Hey! What's up?",
    "how are you": "I'm just a program, but I'm running smoothly! How about you?",
    "what is your name": "I'm RuleBot, a simple rule-based chatbot built for Project 1.",
    "who made you": "I was built by an AI Engineering intern at DecodeLabs!",
    "what can you do": "I can chat with you using pre-defined rules. Try saying hello, bye, or ask my name!",
    "help": "You can talk to me using greetings, ask my name, or type 'bye' to exit.",
    "thank you": "You're welcome!",
    "thanks": "Anytime!",
}

# Words that should end the conversation
exit_commands = {"exit", "quit", "bye", "goodbye", "see you"}

# Fallback replies when nothing matches (adds a bit of personality)
fallback_replies = [
    "I do not understand that yet. Could you rephrase?",
    "Hmm, I'm not trained to answer that. Try 'help' to see what I can do.",
    "Sorry, that's outside my rule set right now.",
]


def get_greeting_by_time():
    """Small bonus: greet differently based on real time of day (nested logic)."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def sanitize(text):
    """PHASE 1: Input Sanitization — lowercase + strip whitespace."""
    return text.lower().strip()


def get_response(user_input):
    """PHASE 2: Process — look up the cleaned input in our knowledge base."""
    # Exact match lookup with fallback, all in one atomic operation
    return responses.get(user_input, random.choice(fallback_replies))


def chat():
    """PHASE 3: The Heartbeat — an infinite loop until the exit command."""
    print(f"RuleBot: {get_greeting_by_time()}! I'm RuleBot 🤖. Type 'bye' anytime to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        # Kill command check
        if clean_input in exit_commands:
            print("RuleBot: Goodbye! Have a great day. 👋")
            break

        # Ignore empty input instead of crashing
        if clean_input == "":
            print("RuleBot: ...I didn't catch that. Say something!")
            continue

        reply = get_response(clean_input)
        print(f"RuleBot: {reply}")


if __name__ == "__main__":
    chat()
