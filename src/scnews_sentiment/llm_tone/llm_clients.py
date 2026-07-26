from __future__ import annotations

import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.scnews_sentiment.llm_tone.utils import clean_generated_text


class BaseLLM(ABC):
    last_response_info: Dict[str, Any] = {}

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class RemoteLLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float,
        top_p: float,
        thinking: str,
        reasoning_effort: str,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.last_response_info = {}
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, max_tokens: int) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful assistant for analyzing and forecasting news events.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort != "none":
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking != "auto":
            request_kwargs["extra_body"] = {"thinking": {"type": self.thinking}}

        response = self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        reasoning_content = getattr(message, "reasoning_content", None)
        self.last_response_info = {
            "backend": "api",
            "model": self.model_name,
            "finish_reason": getattr(choice, "finish_reason", None),
            "raw_content_length": len(content) if content else 0,
            "has_reasoning_content": bool(reasoning_content),
            "reasoning_content_length": len(reasoning_content) if reasoning_content else 0,
        }
        return clean_generated_text(content)


class LocalLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        temperature: float,
        top_p: float,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.temperature = temperature
        self.top_p = top_p
        self.last_response_info = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.torch = torch

    def generate(self, prompt: str, max_tokens: int) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a careful assistant for analyzing and forecasting news events.",
            },
            {"role": "user", "content": prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([input_text], return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )
        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, prompt_len:]
        raw = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.last_response_info = {
            "backend": "local",
            "raw_content_length": len(raw) if raw else 0,
        }
        return clean_generated_text(raw)


def get_llm(args: argparse.Namespace) -> BaseLLM:
    if args.llm_mode == "api":
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"API key not found. Set environment variable {args.api_key_env}, "
                "or change --api-key-env."
            )
        return RemoteLLM(
            api_key=api_key,
            base_url=args.api_base,
            model_name=args.api_model,
            temperature=args.temperature,
            top_p=args.top_p,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
        )
    if args.llm_mode == "local":
        return LocalLLM(
            model_path=args.local_model_path,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    raise ValueError(f"Unsupported --llm-mode: {args.llm_mode}")
