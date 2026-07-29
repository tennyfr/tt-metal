"""One-off probe (push-triggered, not wired into any pipeline).

Re-checks the WAF on bare '/chat/completions' vs '/v1/chat/completions' from a CI
runner, using the real 96 KB sim prompt plus smaller slices — to tell whether the
'/v1' 403 on big payloads is a content rule or a size limit.
"""

import os
import sys

from openai import OpenAI, OpenAIError

base = os.environ["TT_CHAT_URL"].rstrip("/")
key = os.environ["TT_CHAT_API_KEY"]
model = os.environ.get("AI_SUMMARY_MODEL") or os.environ.get("TT_CHAT_MODEL") or "anthropic/claude-sonnet-4-6"

sim = open(os.path.join(os.path.dirname(__file__), "sim_prompt.txt")).read()
payloads = {
    "hello": "hello",
    "dotdot": "../opt/venv/lib/python3.10/site-packages/pydantic/_internal/_config.py:291",
    "sim_16k": sim[:16000],
    "sim_48k": sim[:48000],
    f"sim_full_{len(sim)//1024}k": sim,
}


def status(client, content):
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=16
        )
        return "OK"
    except OpenAIError as e:
        return f"FAIL {getattr(e, 'status_code', '?')}"


for route in ["", "/v1"]:
    c = OpenAI(api_key=key, base_url=base + route)
    label = route or "bare"
    for pname, content in payloads.items():
        print(f"{label:5s} | {pname:12s} ({len(content):6d} ch) -> {status(c, content)}")
        sys.stdout.flush()
