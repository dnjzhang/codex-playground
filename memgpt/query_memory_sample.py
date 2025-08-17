#!/usr/bin/env python
# coding: utf-8
import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import argparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

from letta_client import Letta

client = Letta(base_url="http://localhost:8283")

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


agent_id = "agent-f30d8da9-22fd-4396-bd1c-b9ca36141b11";
memory_block=client.agents.blocks.retrieve(
    agent_id=agent_id,
    block_label="human"
)

print(memory_block)

print("Agent list:")
agents = client.agents.list()          # supports filters like name=..., tags=[...], limit=..., after=...
for a in agents:
    print(f"{a.id}  {a.name}  (created {a.created_at})")

source_id='source-ada61b79-c907-4918-bfe5-75a0904c684e'
passages = client.sources.passages.list(
    source_id=source_id,
)
print(f"Number passages loaded: {len(passages)}")
print(f"Passages: {passages[1]}")



from letta_client import Letta, MessageCreate, TextContent

client = Letta(base_url="http://localhost:8283")

AGENT_ID = "agent-bee6c1a7-0aba-41af-86a4-5185899b092b"  # an ID that shows up in client.agents.list()

resp = client.agents.messages.create(
    agent_id=AGENT_ID,
    messages=[MessageCreate(role="user", content=[TextContent(text="Wake up and say hi!")])]
)
print(resp.messages[-1].content)


