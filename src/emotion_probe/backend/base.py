from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator, TypedDict
import torch


class ReasoningMode(str, Enum):
    NO_THINK  = "no_think"    # empty <think></think> forced before response
    THINK     = "think"       # standard Qwen3 <think>...</think>
    SCRATCHPAD = "scratchpad" # empty think + <SCRATCHPAD_REASONING>...</SCRATCHPAD_REASONING>


class TokenWithEmotions(TypedDict):
    token: str
    section: str              # "think" | "scratchpad" | "response"
    emotions: dict[str, float]


class EmotionProbeBackend(ABC):
    @abstractmethod
    async def generate_with_emotions(
        self,
        user_message: str,
        mode: ReasoningMode,
        emotion_vectors: dict[str, torch.Tensor],
        max_new_tokens: int = 512,
    ) -> AsyncIterator[TokenWithEmotions]:
        """Streaming generation: yield one TokenWithEmotions per token."""
        ...

    @abstractmethod
    def get_layer_for_probing(self) -> int:
        """Return the layer index used for emotion probing."""
        ...
