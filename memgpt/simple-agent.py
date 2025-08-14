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

agent_state = client.agents.create(
    name="simple_agent",
    memory_blocks=[
        {
          "label": "human",
          "value": "My name is Charles",
          "limit": 10000 # character limit
        },
        {
          "label": "persona",
          "value": "You are a helpful assistant and you always use emojis"
        }
    ],
    model="openai/gpt-4o-mini",
    embedding="openai/text-embedding-3-small"
)

response = client.agents.messages.create(
    agent_id=agent_state.id,
    messages=[
        {
            "role": "user",
            "content": "hows it going????"
        }
    ]
)

for message in response.messages:
    print_message(message)

