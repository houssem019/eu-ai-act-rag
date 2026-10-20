"""Bridge Ragas' structured-output LLM interface to our LLMClient, so judge calls go through the same providers
(Claude / Ollama) and the same record/replay cassette as the answers - Ragas runs in CI for free.

Ragas' modern metrics only need `generate(prompt, response_model)` / `agenerate(...)`: the prompt already embeds
the JSON schema and few-shot examples; we return a validated pydantic object.
"""

import asyncio
import json
import re

from pydantic import BaseModel, ValidationError
from ragas.llms.base import InstructorBaseRagasLLM

from aiact.llm.base import LLMClient, Usage

JUDGE_SYSTEM = "You are a meticulous evaluator. Follow the instructions exactly and reply with JSON only."


class JudgeError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    """Models wrap JSON in ``` fences or add a sentence around it; take the outermost {...}."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class RagasJudge(InstructorBaseRagasLLM):
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.usage = Usage()
        self.calls = 0
        self.parse_retries = 0

    def _complete(self, messages: list[dict], schema: dict) -> str:
        resp = self.llm.complete(JUDGE_SYSTEM, messages, json_schema=schema)
        self.calls += 1
        self.usage = self.usage + resp.usage
        return resp.text

    def generate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        schema = response_model.model_json_schema()
        messages = [{"role": "user", "content": prompt}]
        text = self._complete(messages, schema)
        try:
            return response_model.model_validate_json(extract_json(text))
        except ValidationError as e:
            # One repair turn with the validation error, then give up loudly (the sample is skipped, not guessed).
            self.parse_retries += 1
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That reply did not match the required JSON schema: "
                    f"{e.errors(include_url=False)[:3]}. Reply again with only valid JSON matching: {json.dumps(schema)}",
                },
            ]
            text = self._complete(messages, schema)
            try:
                return response_model.model_validate_json(extract_json(text))
            except ValidationError as e2:
                raise JudgeError(f"judge returned invalid {response_model.__name__} twice: {e2}") from e2

    async def agenerate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        return await asyncio.to_thread(self.generate, prompt, response_model)
