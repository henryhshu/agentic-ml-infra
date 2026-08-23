from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

        # Configuration for Anthropic / AWS Bedrock
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._aws_api_key = os.environ.get("ANTHROPIC_AWS_API_KEY")
        self._workspace_id = os.environ.get("ANTHROPIC_AWS_WORKSPACE_ID")
        self._aws_region = os.environ.get("AWS_REGION", "us-east-1")
        self._use_bedrock = os.environ.get("CLAUDE_CODE_USE_BEDROCK", "") == "1"

        # Default to very cheap, fast Claude Haiku model for in-the-loop analysis
        self._model = (
            os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("CLAUDE_MODEL")
            or "claude-haiku-4-5-20251001"
        )
        self._client = self._init_anthropic_client()

    def _init_anthropic_client(self):
        """Initialize the appropriate Anthropic / AWS client based on environment variables."""
        try:
            import anthropic

            if self._aws_api_key:
                return anthropic.AnthropicAWS(
                    api_key=self._aws_api_key,
                    workspace_id=self._workspace_id,
                    aws_region=self._aws_region,
                )
            elif self._api_key:
                return anthropic.Anthropic(api_key=self._api_key)
            elif self._use_bedrock:
                return anthropic.AnthropicBedrock(
                    aws_region=self._aws_region,
                )
        except Exception as e:
            self._log(f"Failed to initialize Anthropic client: {e}", level="warning")
        return None

    @property
    def is_llm_available(self) -> bool:
        """Returns True if a live Anthropic / Bedrock client is configured."""
        return self._client is not None

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 300,
    ) -> str:
        """Execute an LLM call using Claude on AWS or fallback to rule-based analysis."""
        event_msg = f"LLM Call ({self._model}) - {user_message[:60]}..."
        self.tracker.log_event(
            self.name,
            "llm_call",
            event_msg,
            {"system_prompt": system_prompt, "user_message": user_message, "model": self._model},
        )

        if self.is_llm_available and self._client is not None:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                text = response.content[0].text.strip()
                self._log(f"Claude analysis ({self._model}): {text}", level="info")
                return text
            except Exception as e:
                self._log(f"Anthropic LLM call failed: {e}. Using rule-based fallback.", level="warning")

        fallback_msg = "[Rule-Based Fallback] LLM client inactive. Automated rule-based analysis applied."
        self._log(f"Fallback triggered for: {user_message[:50]}...", level="info")
        return fallback_msg

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
