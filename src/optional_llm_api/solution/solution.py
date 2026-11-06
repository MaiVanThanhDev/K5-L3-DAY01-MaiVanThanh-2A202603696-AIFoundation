"""Side lab LLM API; giữ chữ ký và tên khóa tương thích với template.

Đơn giá là giả định của đề, không phải báo giá trực tiếp. Tests dùng mock.
"""

import os
import time
from typing import Any, Callable

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"
COST_PER_1K_OUTPUT_TOKENS = {OPENAI_MODEL: 0.005, OPENAI_MINI_MODEL: 0.0005}


def call_openai(prompt: str, model: str = OPENAI_MODEL,
                temperature: float = 0.7, top_p: float = 0.9,
                max_tokens: int = 256) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none", temperature=temperature, top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or "", time.perf_counter() - start


def call_openai_mini(prompt: str, temperature: float = 0.7, top_p: float = 0.9,
                     max_tokens: int = 256) -> tuple[str, float]:
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)


def compare_models(prompt: str) -> dict:
    large, large_latency = call_openai(prompt)
    small, small_latency = call_openai_mini(prompt)
    return {
        "gpt4o_response": large, "mini_response": small,
        "gpt4o_latency": large_latency, "mini_latency": small_latency,
        "gpt4o_cost_estimate": (len(large.split()) / 0.75 / 1000
                                 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]),
    }


def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    while True:
        try:
            prompt = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue
        messages = history + [{"role": "user", "content": prompt}]
        chunks = []
        print("AI: ", end="", flush=True)
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL, messages=messages, stream=True,
            reasoning_effort="none", max_completion_tokens=256,
        )
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                chunks.append(text)
                print(text, end="", flush=True)
        finally:
            stream.close()
        print()
        history = (messages + [{"role": "assistant", "content": "".join(chunks)}])[-6:]


def retry_with_backoff(fn: Callable, max_retries: int = 3,
                       base_delay: float = 0.1) -> Any:
    """Theo hợp đồng side lab: thử lại mọi Exception tối đa max_retries lần."""
    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries và base_delay phải không âm")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


def batch_compare(prompts: list[str]) -> list[dict]:
    return [{**compare_models(prompt), "prompt": prompt} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    def short(value):
        text = " ".join(str(value).split()).replace("|", "/")
        return text if len(text) <= 40 else text[:37] + "..."

    lines = ["Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency",
             "--- | --- | --- | --- | ---"]
    for row in results:
        lines.append(" | ".join([
            short(row["prompt"]), short(row["gpt4o_response"]),
            short(row["mini_response"]), f"{row['gpt4o_latency']:.3f}s",
            f"{row['mini_latency']:.3f}s",
        ]))
    return "\n".join(lines)


if __name__ == "__main__":
    question = "Explain temperature and top_p in one sentence."
    print(format_comparison_table(batch_compare([question])))
    streaming_chatbot()
