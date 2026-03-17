"""
AI Bootcamp Codebase Agent
An intelligent agent that can analyze and answer questions about the AI bootcamp codebase.
Uses the Claude Agent SDK with built-in file tools.
"""

import anyio
import sys
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage


SYSTEM_PROMPT = """You are an expert AI tutor and code analyst for an AI bootcamp codebase.

The bootcamp teaches machine learning and deep learning from fundamentals to advanced topics:
- Steps 0-5: Math foundations, linear regression, perceptron, logistic regression, neural networks
- Step 6: PyTorch framework
- Steps 7-8: RNNs, LSTMs, Transformers, CNNs, transfer learning, GANs
- Steps 11-13: Reinforcement learning, AI ethics, Graph Neural Networks

When analyzing code, explain concepts clearly for learners at all levels.
When asked questions, search the codebase thoroughly before answering.
Always cite specific files and line numbers when referencing code."""


async def run_agent(prompt: str, working_dir: str = "/home/user/ai-bootcamp") -> str:
    """Run the AI bootcamp agent with a given prompt."""
    session_id = None

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd=working_dir,
            allowed_tools=["Read", "Glob", "Grep"],
            system_prompt=SYSTEM_PROMPT,
            max_turns=20,
        ),
    ):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            session_id = message.data.get("session_id")
            print(f"[Session: {session_id}]", file=sys.stderr)

        if isinstance(message, ResultMessage):
            return message.result

    return ""


async def interactive_session():
    """Run an interactive multi-turn session with the agent."""
    print("AI Bootcamp Codebase Agent")
    print("Type your question about the codebase (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nAgent: ", end="", flush=True)
        result = await run_agent(user_input)
        print(result)
        print()


async def main():
    if len(sys.argv) > 1:
        # Single query mode: pass prompt as command-line argument
        prompt = " ".join(sys.argv[1:])
        print(f"Query: {prompt}\n")
        result = await run_agent(prompt)
        print(result)
    else:
        # Interactive mode
        await interactive_session()


if __name__ == "__main__":
    anyio.run(main)
