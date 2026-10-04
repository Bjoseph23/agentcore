"""
================================================================================
WanderBot — EXERCISE SOLUTION: AgentCore Browser
================================================================================
"""

import logging
import sys

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands_tools.browser import AgentCoreBrowser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WanderBot.AgentCoreBrowser")

# Safety net: caps how deep the agent's tool-use loop can recurse in one
# invocation, since older Strands versions have no built-in turn limit.
sys.setrecursionlimit(5000)

app = BedrockAgentCoreApp()
model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0")

SYSTEM_PROMPT = """You are WanderBot, the AI travel assistant for Horizon Travel.

You have access to a live web browser tool. Use it to look up destination information
on Wikivoyage (en.wikivoyage.org) — a free, open travel guide.

When a customer asks about a destination:
1. Navigate to en.wikivoyage.org/wiki/<DestinationName> ONE time.
2. Read the page ONCE — focus on highlights, neighbourhoods, and practical tips.
3. Summarise what you find clearly and concisely. Do NOT re-navigate to the same
   page, re-read it, or repeat a step you have already completed. If the page
   content is unclear or incomplete, summarise what you got rather than retrying.

Always tell the customer that the information came from Wikivoyage."""


def _build_agent() -> Agent:
    """Construct the Agent, applying a per-invocation turn cap if the
    installed Strands version supports it."""
    browser = AgentCoreBrowser(session_timeout=600)  # 10 min; default is 3600s
    kwargs = dict(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[browser.browser],
    )
    try:
        from strands.types.agent import Limits
        return Agent(**kwargs, limits=Limits(max_turns=15))
    except (ImportError, TypeError):
        # Limits either isn't available in this version, or Agent.__init__()
        # doesn't accept it yet — fall back to the recursion ceiling + prompt.
        return Agent(**kwargs)


@app.entrypoint
async def invoke(payload: dict, context=None) -> dict:
    """WanderBot — AgentCore Browser entry point."""
    user_message = payload.get("message", "Hello!")
    logger.info("User: %s", user_message[:80])

    agent = _build_agent()

    try:
        return agent(user_message)
    except Exception as e:
        logger.error("Agent failed. Message history: %s", getattr(agent, "messages", None))
        logger.exception("Agent invocation raised: %s", e)
        raise


if __name__ == "__main__":
    app.run()