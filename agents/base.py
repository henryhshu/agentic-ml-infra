from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from metrics.experiment import ExperimentTracker
    from pipeline.config import PipelineConfig
    from agents.checkpoints import CheckpointEmitter

class AgentBase(ABC):
    def __init__(
        self, 
        name: str, 
        tracker: ExperimentTracker, 
        config: PipelineConfig, 
        checkpoint_emitter: CheckpointEmitter | None = None
    ) -> None:
        self.name = name
        self.tracker = tracker
        self.config = config
        self.checkpoint_emitter = checkpoint_emitter
        
        self._api_key = os.environ.get('ANTHROPIC_API_KEY')
        self._use_bedrock = os.environ.get('CLAUDE_CODE_USE_BEDROCK', '') == '1'

    @property
    def is_llm_available(self) -> bool:
        return bool(self._api_key) or self._use_bedrock

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        event_msg = f"LLM Call - System: {system_prompt[:50]}... User: {user_message[:50]}..."
        self.tracker.log_event(self.name, "llm_call", event_msg, {"system_prompt": system_prompt, "user_message": user_message})
        
        if self.is_llm_available:
            try:
                import claude_agent_sdk  # type: ignore
                return "Mock LLM Response"
            except ImportError:
                self._log("claude_agent_sdk not found, using fallback.", level="warning")
                
        fallback_msg = "[LLM_PLACEHOLDER] No API key configured. Rule-based fallback used."
        self._log(f"Fallback LLM triggered for prompt: {user_message[:50]}...", level="info")
        return fallback_msg

    @abstractmethod
    def observe(self, event: dict[str, Any]) -> list[str]:
        pass

    @abstractmethod
    def diagnose(self, anomalies: list[str], context: dict[str, Any]) -> str:
        pass

    @abstractmethod
    def recommend_action(self, diagnosis: str, context: dict[str, Any]) -> str | None:
        pass

    def _log(self, message: str, level: str = 'info') -> None:
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
