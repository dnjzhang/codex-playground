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

# Attach a source
source = client.sources.create(
    name="pltr-10k-4",
    embedding="openai/text-embedding-3-small"
)

job = client.sources.files.upload(
    source_id=source.id,
    file=open("PLTR-FY2024-10-K.pdf", "rb")
)
print(f">>{job}")

source = client.sources.retrieve(source.id)
print(f"Source: {source}")

# If status is "ready", proceed with using the source
#if source_status.status == "ready":
    # Continue with your code...
#    passages = client.sources.passages.list(
#        source_id=source.id,
#    )
#    print(f"Number passages loaded: {len(passages)}")

print("Data source loaded with source ID: " + source.id + "")

passages = client.sources.passages.list(
    source_id=source.id,
)
print(f"Number passages loaded: {len(passages)}")

# Attach the data source to the agent
agent_state = client.agents.create(
    name="demo_rag_agent",
    memory_blocks=[
        {
          "label": "human",
          "value": "My name is John",
          "limit": 10000 # character limit
        },
        {
          "label": "persona",
          "value": "You are a helpful financial assistant who is well-versed to review public company financial reports."
        }
    ],
    model="openai/gpt-4o-mini",
    embedding="openai/text-embedding-3-small"
)

agent_state = client.agents.sources.attach(
    agent_id=agent_state.id,
    source_id=source.id
)

print("Agent ID: " + agent_state.id)
response = client.agents.messages.create(
    agent_id=agent_state.id,
    messages=[
        {
            "role": "user",
            "content": "What are the top 3 revenue products of Palantir Technologies in 2024?"
        }
    ]
)

for message in response.messages:
    print_message(message)

