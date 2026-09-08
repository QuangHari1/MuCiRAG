"""OpenAI-compatible LLM calls and pluggable embedding calls for RAG."""

from __future__ import annotations

import json
import re
from typing import Protocol

from MuCiRAG.src.embeddings import EmbeddingProvider
from MuCiRAG.src.settings import require_secret


RELEASE_TAG_PATTERN = re.compile(r"\s*\[\s*3GPP\s+Release\s+\d+\s*\]\s*", re.IGNORECASE)
RELEASE_MENTION_PATTERN = re.compile(
    r"\b(?:according\s+to\s+|under\s+|in\s+|for\s+|from\s+|within\s+)?"
    r"(?:the\s+)?(?:3GPP\s+)?Release\s+\d+\b",
    re.IGNORECASE,
)
TRAILING_METADATA_PATTERN = re.compile(
    r"(?:\b(?:according\s+to|under|in|for|from|within|of|the|context\s+of|reference\s+to)\b\s*)+$",
    re.IGNORECASE,
)


class RagClient(Protocol):
    def rephrase(self, question: str) -> str: ...

    def extract_facets(self, question: str) -> list[str]: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...

    def answer(
        self, question: str, contexts: list[str], strict_multiple_choice: bool = False
    ) -> str: ...


class OpenAICompatibleRagClient:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        rephrase_model: str,
        answer_model: str,
        llm_provider: str,
        api_key_env: str,
        base_url: str,
        thinking_mode: str,
        temperature: float,
    ):
        from openai import OpenAI

        if llm_provider not in {"openai", "deepseek", "openrouter"}:
            raise ValueError(f"Unsupported [llm].provider: {llm_provider}")
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("[llm].thinking_mode must be 'enabled' or 'disabled'.")
        if not 0 <= temperature <= 2:
            raise ValueError("[llm].temperature must be between 0 and 2.")
        client_args = {"api_key": require_secret(api_key_env)}
        if base_url:
            client_args["base_url"] = base_url
        self._client = OpenAI(**client_args)
        self.embedding_provider = embedding_provider
        self._rephrase_model = rephrase_model
        self._answer_model = answer_model
        self._llm_provider = llm_provider
        self._thinking_mode = thinking_mode
        self._temperature = temperature

    @staticmethod
    def _uses_gpt_5_6(model: str) -> bool:
        """Identify OpenAI's GPT-5.6 family, which fixes temperature at its default."""
        return model.startswith("gpt-5.6")

    def _completion_options(self, model: str) -> dict[str, object]:
        uses_gpt_5_6 = self._llm_provider == "openai" and self._uses_gpt_5_6(model)
        # GPT-5.6 rejects an explicit temperature other than its default of 1.
        options: dict[str, object] = {} if uses_gpt_5_6 else {"temperature": self._temperature}
        if uses_gpt_5_6:
            options["reasoning_effort"] = "medium" if self._thinking_mode == "enabled" else "none"
        if self._llm_provider == "deepseek":
            options["extra_body"] = {"thinking": {"type": self._thinking_mode}}
        elif self._llm_provider == "openrouter":
            options["extra_body"] = {
                "reasoning": {"enabled": self._thinking_mode == "enabled"}
            }
        return options

    def rephrase(self, question: str) -> str:
        """Keep the original rephrase-only retrieval prompt."""
        response = self._client.chat.completions.create(
            model=self._rephrase_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Rephrase the question to be clear and concise:\n\n{question}",
                }
            ],
            **self._completion_options(self._rephrase_model),
        )
        return (response.choices[0].message.content or question).strip()

    def extract_facets(self, question: str) -> list[str]:
        """Extract citation-scoring needs without changing the retrieval query."""
        clean_question = RELEASE_TAG_PATTERN.sub(" ", question).strip()
        response = self._client.chat.completions.create(
            model=self._rephrase_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract answer-bearing needs only for scoring evidence cited by retrieved chunks. "
                        "These facets will not be used as a retrieval query. Return one JSON object with exactly:\n"
                        '- "facets": the smallest list of information needs required to answer the question.\n\n'
                        "A facet must describe what the answer must establish, including the relevant entity, "
                        "relation, condition, action, or quantity. Do not output isolated entities, acronyms, "
                        "broad topic labels, release numbers, document names, standards, provenance, or phrases "
                        "such as 'according to 3GPP'. Return no more than four facets. If the question asks for "
                        "one relation, property, condition, action, or quantity, return exactly one facet. "
                        "Do not answer the question and do not invent information.\n\n"
                        f"Question:\n{clean_question}"
                    ),
                }
            ],
            response_format={"type": "json_object"},
            **self._completion_options(self._rephrase_model),
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return [clean_question]
        if not isinstance(payload, dict):
            return [clean_question]
        raw_facets = payload.get("facets")
        facets: list[str] = []
        if isinstance(raw_facets, list):
            for raw_facet in raw_facets:
                if not isinstance(raw_facet, str):
                    continue
                facet = RELEASE_TAG_PATTERN.sub(" ", raw_facet)
                facet = RELEASE_MENTION_PATTERN.sub(" ", facet)
                facet = TRAILING_METADATA_PATTERN.sub("", facet)
                facet = re.sub(r"\s+", " ", facet).strip(" ,;:-")
                if facet and facet not in facets:
                    facets.append(facet)
                if len(facets) == 4:
                    break
        if not facets:
            facets = [clean_question]
        return facets

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_provider.embed(texts)

    def answer(
        self, question: str, contexts: list[str], strict_multiple_choice: bool = False
    ) -> str:
        context = "\n\n".join(contexts)
        if strict_multiple_choice:
            instruction = (
                "Choose the correct option using only the retrieved 3GPP context. Your entire response is "
                "machine-scored: return exactly one plain-text label `Option N`, with N replaced by one listed "
                "option number. Output nothing else: no explanation, reasoning, punctuation, Markdown, code "
                "fence, option text, or second option."
            )
        else:
            instruction = (
                "Answer the question using only the retrieved 3GPP context when it is relevant. "
                "State uncertainty when the context is insufficient."
            )
        request_options = self._completion_options(self._answer_model)
        if strict_multiple_choice:
            token_limit_name = "max_completion_tokens" if self._uses_gpt_5_6(self._answer_model) else "max_tokens"
            request_options[token_limit_name] = 8
        response = self._client.chat.completions.create(
            model=self._answer_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{instruction}\n\n"
                        f"Question:\n{question}\n\nRetrieved context:\n{context}"
                    ),
                }
            ],
            **request_options,
        )
        return (response.choices[0].message.content or "").strip()
