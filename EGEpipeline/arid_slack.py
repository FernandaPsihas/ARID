"""ARID Slack bot -- @-mention it in a channel, it replies in-thread with a
grounded, cited dunereco answer. Wraps answer.py's answer(), same retrieval +
local qwen3-coder:30b every other entry point uses.

Socket mode (not HTTP): Slack opens the connection outbound to us over a
WebSocket, so this runs on the Tailscale-internal host with NO public URL,
ingress, or reverse proxy. That's why it needs an APP-level token (xapp-, for
the socket) in addition to the bot token (xoxb-).

Setup (one-time, see the block comment at the bottom for the full checklist):
    .venv/bin/pip install -r requirements-slack.txt
    export SLACK_BOT_TOKEN=xoxb-...   # Bot User OAuth Token
    export SLACK_APP_TOKEN=xapp-...   # App-Level Token, scope connections:write
    python EGEpipeline/arid_slack.py

Smoke test without Slack:  python EGEpipeline/arid_slack.py --test
"""

import os
import re
import sys

# venv guard first (same as answer.py/chat.py) -- gets us qdrant-client + ollama.
sys.path.insert(0, os.path.dirname(__file__))
from env_setup import ensure_env
ensure_env()

from answer import answer  # returns the full cited answer string (body + Sources)


def _strip_mention(text: str) -> str:
    """Drop the '<@U123>' bot mention (and any other user mentions) Slack puts in
    the event text, leaving just the question."""
    return re.sub(r"<@[^>]+>", "", text).strip()


def _build_app():
    """Import slack_bolt lazily so --test and the self-check don't need it."""
    try:
        from slack_bolt import App
    except ModuleNotFoundError as e:
        sys.exit("arid_slack: slack-bolt isn't installed in this venv -- run "
                 "`.venv/bin/pip install -r requirements-slack.txt`.")

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not bot_token:
        sys.exit("arid_slack: set SLACK_BOT_TOKEN (xoxb-...) -- see setup in the module docstring.")

    app = App(token=bot_token)

    @app.event("app_mention")
    def on_mention(event, say):
        question = _strip_mention(event.get("text", ""))
        thread_ts = event.get("thread_ts") or event["ts"]  # reply in-thread
        if not question:
            say(text="Ask me something about the dunereco codebase, e.g. "
                     "`@arid how is neutrino energy reconstructed?`", thread_ts=thread_ts)
            return
        try:
            reply = answer(question)  # ~5s on GPU; socket mode already ack'd the event
        except Exception as e:  # never leave the mention hanging silently
            reply = f"Sorry, something broke answering that ({type(e).__name__}: {e})."
        say(text=reply, thread_ts=thread_ts)

    return app


def _selfcheck():
    assert _strip_mention("<@U0ARID> how is energy reconstructed?") == "how is energy reconstructed?"
    assert _strip_mention("<@U0ARID>   spaced   ") == "spaced"
    assert _strip_mention("<@U0A> ping <@U0B> pong") == "ping  pong"
    assert _strip_mention("<@U0ARID>") == ""
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _selfcheck()
        sys.exit(0)

    import logging
    logging.basicConfig(level=logging.INFO)  # so the socket connect/disconnect events are visible

    from slack_bolt.adapter.socket_mode import SocketModeHandler
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not app_token:
        sys.exit("arid_slack: set SLACK_APP_TOKEN (xapp-...) for socket mode -- see module docstring.")

    app = _build_app()
    print("arid_slack: connecting to Slack (socket mode)...", file=sys.stderr)
    SocketModeHandler(app, app_token).start()

# ---------------------------------------------------------------------------
# One-time Slack app setup (api.slack.com/apps -> Create New App -> from scratch):
#   1. Socket Mode: ON  (generates the xapp- App-Level Token; scope connections:write)
#   2. OAuth & Permissions -> Bot Token Scopes: app_mentions:read, chat:write
#   3. Event Subscriptions: ON, subscribe to bot event `app_mention`
#   4. Install to Workspace -> copy the Bot User OAuth Token (xoxb-)
#   5. Create the private channel, /invite @arid into it
# No public URL / manifest hosting needed -- socket mode dials out from here.
# ---------------------------------------------------------------------------
