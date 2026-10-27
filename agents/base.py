from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from dotenv import load_dotenv

# Load environment variables from .env file (always override existing shell env vars)
load_dotenv(override=True)

if TYPE_CHECKING:
    from metrics.experiment import ExperimentTracker
    from pipeline.config import PipelineConfig
    from agents.checkpoints import CheckpointEmitter


class AgentBase(ABC):
    """Abstract base class for all pipeline monitoring and remediation agents."""

    def __init__(
        self,
        name: str,
        tracker: ExperimentTracker,
        config: PipelineConfig,
        checkpoint_emitter: CheckpointEmitter | None = None,
    ) -> None:
        self.name = name
        self.tracker = tracker
        self.config = config
        self.checkpoint_emitter = checkpoint_emitter

        # Configuration for Claude Platform on AWS
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._aws_api_key = os.environ.get("ANTHROPIC_AWS_API_KEY")
        self._aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        self._aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self._workspace_id = os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")
        self._aws_region = os.environ.get("AWS_REGION", "us-east-1")

        # Default to fast Claude Haiku model for in-the-loop analysis
        self._model = (
            os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("CLAUDE_MODEL")
            or "claude-haiku-4-5-20251001"
        )
        self._client = self._init_anthropic_client()

    def _init_anthropic_client(self):
        """Initialize the Anthropic client for Claude Platform on AWS."""
        try:
            from anthropic import AnthropicAWS

            if self._aws_access_key and self._aws_secret_key:
                return AnthropicAWS(
                    aws_access_key=self._aws_access_key,
                    aws_secret_key=self._aws_secret_key,
                    workspace_id=self._workspace_id,
                    aws_region=self._aws_region,
                )
            elif self._aws_api_key:
                return AnthropicAWS(
                    api_key=self._aws_api_key,
                    workspace_id=self._workspace_id,
                    aws_region=self._aws_region,
                )
            elif self._api_key:
                import anthropic
                return anthropic.Anthropic(api_key=self._api_key)
            else:
                self._log(
                    "No Claude Platform on AWS credentials configured (AWS_ACCESS_KEY_ID or ANTHROPIC_AWS_API_KEY missing).",
                    level="warning",
                )
                return None
        except Exception as e:
            self._log(f"Failed to initialize Anthropic client: {e}", level="error")
            raise RuntimeError(f"Failed to initialize Claude Platform on AWS client: {e}") from e

    @property
    def is_llm_available(self) -> bool:
        """Returns True if a live Claude Platform on AWS client is configured."""
        return self._client is not None

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
    ) -> str:
        """Execute an LLM call using Claude Platform on AWS.

        Rule-based fallbacks are strictly disabled; any failure or missing client
        will raise an exception to validate LLM agent capabilities.
        """
        if not self.is_llm_available or self._client is None:
            raise RuntimeError(
                f"[{self.name}] LLM client is not configured. Rule-based fallback is disabled; "
                "an active Claude Platform on AWS configuration (ANTHROPIC_AWS_API_KEY) is required."
            )

        event_msg = f"LLM Call ({self._model}) - {user_message[:60]}..."
        self.tracker.log_event(
            self.name,
            "llm_call",
            event_msg,
            {"system_prompt": system_prompt, "user_message": user_message, "model": self._model},
        )

        caller = sys._getframe(1).f_code.co_name
        t_start = time.time()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            t_end = time.time()
            usage = getattr(response, "usage", None)
            llm_calls = getattr(self.tracker, "llm_calls", None)
            if llm_calls is not None:
                llm_calls.append({
                    "agent": self.name,
                    "caller": caller,
                    "model": self._model,
                    "start": t_start,
                    "end": t_end,
                    "duration_s": t_end - t_start,
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                })
            text = response.content[0].text.strip()
            self._log(f"Claude analysis ({self._model}): {text}", level="info")
            return text
        except Exception as e:
            self._log(f"Claude Platform on AWS LLM call failed: {e}", level="error")
            raise RuntimeError(
                f"[{self.name}] Claude Platform on AWS LLM call failed: {e}. "
                "Rule-based fallback is disabled."
            ) from e

    @abstractmethod
    def observe(self, event: dict[str, Any]) -> list[str]:
        """Observe incoming telemetry and return detected anomaly types."""
        pass

    @abstractmethod
    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        """Diagnose root cause for detected anomalies."""
        pass

    @abstractmethod
    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        """Recommend a remediation action based on diagnosis."""
        pass

    def _log(self, message: str, level: str = "info") -> None:
        """Log message with an emoji prefix and record to ExperimentTracker."""
        emoji = "ℹ️"
        if level == "warning":
            emoji = "⚠️"
        elif level == "error":
            emoji = "❌"
        elif level == "success":
            emoji = "✅"
        else:
            emoji = "🤖"

        formatted_message = f"{emoji} [{self.name}] {message}"
        print(formatted_message)
        self.tracker.log_event(self.name, f"agent_log_{level}", message)
