"""CLI entrypoint for the local LangChain chatbot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import AppSettings
from app.runtime import AgentRuntime, project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local LangChain DeepAgent chatbot")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use for the initial session",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and print only the final response",
    )
    return parser


def print_help() -> None:
    print("Commands:")
    print("  /help                 Show this help")
    print("  /new                  Start a new session with the current model")
    print("  /model <name>         Start a new session with a different model")
    print("  /session              Show the current session ID and model")
    print("  /quit                 Exit")


def _clear_indicator_line() -> None:
    if sys.stdout.isatty():
        print("\r\033[K", end="", flush=True)


async def thinking_indicator(done: asyncio.Event, label_ref: list[str]) -> None:
    if not sys.stdout.isatty():
        await done.wait()
        return
    frames = ("|", "/", "-", "\\")
    index = 0
    while not done.is_set():
        label = label_ref[0] if label_ref else "thinking..."
        print(f"\r{frames[index % len(frames)]} {label}", end="", flush=True)
        index += 1
        try:
            await asyncio.wait_for(done.wait(), timeout=0.12)
        except asyncio.TimeoutError:
            continue
    print("\r", end="", flush=True)


async def stop_indicator(done: asyncio.Event, task: asyncio.Task[None]) -> None:
    done.set()
    await task
    _clear_indicator_line()


def print_status_line(text: str) -> None:
    _clear_indicator_line()
    print(f"status: {text}")


def next_spinner_label(status_text: str) -> str:
    if status_text.startswith("tool error"):
        return "recovering from tool error..."
    if status_text.startswith("tool "):
        return "running tool..."
    if status_text.startswith("skill "):
        return "using skill..."
    return "thinking..."


async def interactive_chat(model_name: str | None, stream: bool) -> None:
    settings = AppSettings.load(project_root())
    runtime = AgentRuntime(settings)
    await runtime.initialize()

    session = await runtime.create_session(model_name)
    print(f"Session: {session.session_id}")
    print(f"Model: {session.model_name}")
    print("Type /help for commands.")

    try:
        while True:
            try:
                user_text = input("\n$> ").strip()
            except EOFError:
                print()
                break

            if not user_text:
                continue
            if user_text == "/quit":
                break
            if user_text == "/help":
                print_help()
                continue
            if user_text == "/new":
                await runtime.close_session(session.session_id)
                session = await runtime.create_session(session.model_name)
                print(f"Session: {session.session_id}")
                print(f"Model: {session.model_name}")
                continue
            if user_text.startswith("/model "):
                next_model = user_text.split(maxsplit=1)[1].strip()
                await runtime.close_session(session.session_id)
                session = await runtime.create_session(next_model)
                print(f"Session: {session.session_id}")
                print(f"Model: {session.model_name}")
                continue
            if user_text == "/session":
                print(f"Session: {session.session_id}")
                print(f"Model: {session.model_name}")
                continue

            spinner_done = asyncio.Event()
            spinner_label = ["thinking..."]
            spinner_task = asyncio.create_task(thinking_indicator(spinner_done, spinner_label))
            try:
                use_deep_agent = runtime.should_use_deep_agent(user_text)
                if use_deep_agent:
                    print_status_line("thinking")
                    printed_status = False
                    async for kind, text in runtime.stream_deep_agent_updates(
                        session.session_id,
                        user_text,
                    ):
                        if kind == "status":
                            print_status_line(text)
                            spinner_label[0] = next_spinner_label(text)
                            printed_status = True
                        elif kind == "text":
                            if not spinner_done.is_set():
                                await stop_indicator(spinner_done, spinner_task)
                            if printed_status:
                                print()
                            print(text)
                    continue

                if stream:
                    printed = False
                    async for text in runtime.stream_chat(session.session_id, user_text):
                        if not spinner_done.is_set():
                            await stop_indicator(spinner_done, spinner_task)
                        print(text, end="", flush=True)
                        printed = True
                    if not printed:
                        await stop_indicator(spinner_done, spinner_task)
                        text = await runtime.final_response_text(session.session_id)
                        print(text, end="", flush=True)
                    elif not spinner_done.is_set():
                        await stop_indicator(spinner_done, spinner_task)
                    print()
                    continue

                response = await runtime.chat_once(session.session_id, user_text)
                await stop_indicator(spinner_done, spinner_task)
                print(response)
            except Exception as exc:  # noqa: BLE001
                if not spinner_done.is_set():
                    await stop_indicator(spinner_done, spinner_task)
                logging.exception("Request failed")
                print(f"Runtime error: {exc}")
    finally:
        await runtime.aclose()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args()
    await interactive_chat(args.model, not args.no_stream)


if __name__ == "__main__":
    asyncio.run(main())
