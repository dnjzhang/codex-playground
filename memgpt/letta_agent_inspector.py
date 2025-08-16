#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from letta_client import Letta

# ---------- Helpers ----------

def fmt_dt(dt):
    try:
        if isinstance(dt, str):
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S %Z")
        return str(dt)
    except Exception:
        return str(dt)

def print_agent_row(a):
    created = getattr(a, "created_at", None)
    print(f"{getattr(a, 'id', '')}\t{getattr(a, 'name', '')}\t(created {fmt_dt(created)})")

def safe_block_label(b):
    return getattr(b, "label", getattr(b, "block_label", "unknown"))

def safe_block_content(b):
    for attr in ("content", "text", "value", "data"):
        if hasattr(b, attr) and getattr(b, attr) is not None:
            return getattr(b, attr)
    return repr(b)

def dump_known_blocks(client: Letta, agent_id: str):
    print("\nMemory blocks:")
    tried_any = False
    try:
        blocks = client.agents.blocks.list(agent_id=agent_id)  # if supported
        if blocks:
            for b in blocks:
                label = safe_block_label(b)
                print(f"\n[{label}]")
                print(safe_block_content(b))
                tried_any = True
    except Exception:
        pass
    if not tried_any:
        for lbl in ["human", "system", "tools", "tasks", "short_term", "long_term", "profile"]:
            try:
                b = client.agents.blocks.retrieve(agent_id=agent_id, block_label=lbl)
                if b:
                    print(f"\n[{lbl}]")
                    print(safe_block_content(b))
                    tried_any = True
            except Exception:
                continue
    if not tried_any:
        print("(no blocks found or this SDK/server doesn’t expose block listing)")

# ---------- Commands ----------

def cmd_list(client: Letta, _args):
    try:
        agents = client.agents.list()
    except Exception as e:
        print(f"Error listing agents: {e}", file=sys.stderr)
        sys.exit(1)
    if not agents:
        print("No agents found.")
        return
    for a in agents:
        print_agent_row(a)

def cmd_dump(client: Letta, args):
    agent_id = args.agent_id
    agent = None
    direct_error = None
    for method_name in ("retrieve", "get"):
        try:
            method = getattr(client.agents, method_name, None)
            if method:
                agent = method(agent_id=agent_id)
                break
        except Exception as e:
            direct_error = e
    if agent is None:
        try:
            agents = client.agents.list()
            agent = next((a for a in agents if getattr(a, "id", None) == agent_id), None)
        except Exception:
            pass
    if agent is None:
        msg = f"Agent '{agent_id}' not found."
        if direct_error:
            msg += f" (retrieve error: {direct_error})"
        print(msg, file=sys.stderr)
        sys.exit(2)

    print("Agent details:")
    print(f"ID:          {getattr(agent, 'id', '')}")
    print(f"Name:        {getattr(agent, 'name', '')}")
    print(f"Description: {getattr(agent, 'description', '')}")
    print(f"Tags:        {getattr(agent, 'tags', [])}")
    print(f"Created:     {fmt_dt(getattr(agent, 'created_at', ''))}")
    print(f"Updated:     {fmt_dt(getattr(agent, 'updated_at', ''))}")

    dump_known_blocks(client, agent_id)

# ---------- Main ----------

def main():
    load_dotenv()  # optional; lets you override base URL via env if you want
    base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
    client = Letta(base_url=base_url)  # no token

    parser = argparse.ArgumentParser(description="Letta Agent Inspector (local, no auth)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all agents")
    p_list.set_defaults(func=lambda a: cmd_list(client, a))

    p_dump = sub.add_parser("dump", help="Dump details (and memory blocks) for an agent")
    p_dump.add_argument("agent_id", help="Agent ID to dump")
    p_dump.set_defaults(func=lambda a: cmd_dump(client, a))

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
