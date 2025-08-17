#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
from typing import Any, List

from letta_client import Letta

# Try to support both "typed" and "dict" payloads depending on SDK version
MessageCreate = None
TextContent = None
for mod in ("letta_client", "letta_client.types", "letta_client.schemas"):
    try:
        m = __import__(mod, fromlist=["MessageCreate", "TextContent"])
        MessageCreate = getattr(m, "MessageCreate", MessageCreate)
        TextContent = getattr(m, "TextContent", TextContent)
    except Exception:
        pass


def parse_source_ids(values: List[str]) -> List[str]:
    """Accept repeated --source-id and/or comma-separated lists."""
    out: List[str] = []
    for v in values or []:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    # de-dupe but preserve order
    seen = set()
    unique = []
    for sid in out:
        if sid not in seen:
            seen.add(sid)
            unique.append(sid)
    return unique


def make_user_message(text: str, reference_items: List[Any] = None) -> Any:
    """Return a user message, optionally with reference content items, for your SDK."""
    reference_items = reference_items or []
    if MessageCreate and TextContent:
        return MessageCreate(
            role="user",
            content=[TextContent(text=text)] + reference_items
        )
    # Fallback: raw dict payload
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}] + reference_items,
    }


def as_reference_content_items(source_ids: List[str]) -> List[Any]:
    """Construct best-effort reference items for message content."""
    items = []
    for sid in source_ids:
        # Common shape used by many toolkits; harmless if server ignores it
        items.append({"type": "reference", "reference": {"source_id": sid}})
    return items


def attach_sources_to_agent(client: Letta, agent_id: str, source_ids: List[str]) -> str:
    """
    Try multiple SDK shapes to attach/link sources to an agent.
    Returns a short string describing what worked, or '' if nothing worked.
    """
    if not source_ids:
        return ""

    # 1) agents.sources.attach(agent_id=..., source_ids=[...])
    try:
        if hasattr(client.agents, "sources") and hasattr(client.agents.sources, "attach"):
            client.agents.sources.attach(agent_id=agent_id, source_ids=source_ids)
            return "agents.sources.attach(list)"
    except Exception:
        pass

    # 2) agents.sources.add / link / create per source_id
    try:
        if hasattr(client.agents, "sources"):
            for mname in ("add", "link", "create", "attach"):
                meth = getattr(client.agents.sources, mname, None)
                if callable(meth):
                    worked_any = False
                    for sid in source_ids:
                        try:
                            meth(agent_id=agent_id, source_id=sid)
                            worked_any = True
                        except Exception:
                            continue
                    if worked_any:
                        return f"agents.sources.{mname}(single)"
    except Exception:
        pass

    # 3) agents.modify/update(..., <various param names>=list)
    for mname in ("modify", "update"):
        meth = getattr(client.agents, mname, None)
        if callable(meth):
            for param in ("source_ids", "knowledge_source_ids", "attached_source_ids", "sources"):
                try:
                    kwargs = {"agent_id": agent_id, param: source_ids}
                    meth(**kwargs)
                    return f"agents.{mname}({param})"
                except Exception:
                    continue

    # 4) agents.references / refs.create(...)
    for path in ("references", "refs"):
        sub = getattr(client.agents, path, None)
        if sub is not None:
            for mname in ("create", "add", "link"):
                meth = getattr(sub, mname, None)
                if callable(meth):
                    worked_any = False
                    for sid in source_ids:
                        try:
                            meth(agent_id=agent_id, source_id=sid)
                            worked_any = True
                        except Exception:
                            continue
                    if worked_any:
                        return f"agents.{path}.{mname}(single)"

    return ""  # nothing worked


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if hasattr(c, "text") and getattr(c, "text") is not None:
                parts.append(getattr(c, "text"))
            elif isinstance(c, dict):
                if "text" in c and c["text"] is not None:
                    parts.append(str(c["text"]))
                elif "value" in c and c["value"] is not None:
                    parts.append(str(c["value"]))
        return "\n".join(p for p in parts if p)
    if hasattr(content, "text") and getattr(content, "text") is not None:
        return getattr(content, "text")
    return str(content)


def print_message(message):
    if message.message_type == "reasoning_message":
        print("🧠 Reasoning: " + message.reasoning)
    elif message.message_type == "assistant_message":
        print("🤖 Agent: " + message.content)
    elif message.message_type == "tool_call_message":
        print("🔧 Tool Call: " + message.tool_call.name + "\n" + message.tool_call.arguments)
    elif message.message_type == "tool_return_message":
        print("🔧 Tool Return: " + message.tool_return)
    elif message.message_type == "user_message":
        print("👤 User Message: " + message.content)

def print_last_assistant(resp: Any) -> int:
    msgs = getattr(resp, "messages", None)
    if msgs is None and isinstance(resp, dict):
        msgs = resp.get("messages")
    if not msgs:
        print(resp)
        return 1
    last_assistant = None
    for m in msgs[::-1]:
        if m.message_type == "assistant_message":
            last_assistant = m
            break
    if last_assistant is None:
        last = msgs[-1]
        role = getattr(last, "role", None) or (last.get("role") if isinstance(last, dict) else "unknown")
        content = getattr(last, "content", None) if not isinstance(last, dict) else last.get("content")
        print(f"[{role}] {extract_text_from_content(content)}")
        return 0
    content = getattr(last_assistant, "content", None) if not isinstance(last_assistant, dict) else last_assistant.get("content")
    text = extract_text_from_content(content)
    print(text.strip() or last_assistant)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Send a message to a Letta agent, optionally attaching source IDs for references.")
    parser.add_argument("agent_id", help="Agent ID (e.g., agent-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
    parser.add_argument("-m", "--message", default="Wake up and say hi!", help="Message to send (default: %(default)s)")
    parser.add_argument("-u", "--url", default=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
                        help="Letta server base URL (default: %(default)s)")
    parser.add_argument("--source-id", action="append",
                        help="Attach a source ID to the agent (repeatable) or pass comma-separated list")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose status output")
    args = parser.parse_args()

    src_ids = parse_source_ids(args.source_id or [])

    try:
        client = Letta(base_url=args.url)

        # Best effort: attach sources to the agent (covers multiple SDKs)
        attach_used = attach_sources_to_agent(client, args.agent_id, src_ids)
        if args.verbose and attach_used:
            print(f"[info] Attached sources via: {attach_used}")

        # Build message payload. If we couldn't attach, try embedding as reference items.
        reference_items = []
        if src_ids and not attach_used:
            reference_items = as_reference_content_items(src_ids)

        payload = [make_user_message(args.message, reference_items)]

        try:
            resp = client.agents.messages.create(agent_id=args.agent_id, messages=payload)
        except Exception as e:
            # If embedding reference items caused a schema error, fall back to plain text with refs appended.
            if src_ids and reference_items:
                fallback_text = args.message + "\n\nReferences: " + ", ".join(src_ids)
                payload = [make_user_message(fallback_text, [])]
                resp = client.agents.messages.create(agent_id=args.agent_id, messages=payload)
            else:
                raise e

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    rc = print_last_assistant(resp)
    sys.exit(rc)


if __name__ == "__main__":
    main()
