"""One-off probe (push-triggered, not wired into any pipeline).

Tests whether the gateway WAF is scoped to the '/chat/completions' path: sends a
known-403 payload ('../') to both base_url + '/chat/completions' (what the client
does today) and base_url + '/v1/chat/completions'. If '/v1' returns 200 while the
bare path 403s, switching the endpoint bypasses the WAF wholesale.
"""

import os
import sys

from openai import OpenAI, OpenAIError

base = os.environ["TT_CHAT_URL"].rstrip("/")
key = os.environ["TT_CHAT_API_KEY"]
model = os.environ.get("AI_SUMMARY_MODEL") or os.environ.get("TT_CHAT_MODEL") or "gpt-4o"

clients = {
    "bare  (.../chat/completions)": OpenAI(api_key=key, base_url=base),
    "v1    (.../v1/chat/completions)": OpenAI(api_key=key, base_url=base + "/v1"),
}
payloads = {
    "hello": "hello",
    "waf_trigger_../": "../opt/venv/lib/python3.10/site-packages/pydantic/_internal/_config.py:291",
}


def status(client, content):
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=16
        )
        return "OK"
    except OpenAIError as e:
        return f"FAIL {getattr(e, 'status_code', '?')}"


for cname, client in clients.items():
    for pname, content in payloads.items():
        print(f"{cname:34s} | {pname:16s} -> {status(client, content)}")
        sys.stdout.flush()
