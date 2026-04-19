"""Phase 2 stub: Modal + vLLM backend (not implemented)."""

import torch
from typing import AsyncIterator
from emotion_probe.backend.base import EmotionProbeBackend, ReasoningMode, TokenWithEmotions


class ModalVLLMBackend(EmotionProbeBackend):
    async def generate_with_emotions(
        self,
        user_message: str,
        mode: ReasoningMode,
        emotion_vectors: dict[str, torch.Tensor],
        max_new_tokens: int = 512,
    ) -> AsyncIterator[TokenWithEmotions]:
        raise NotImplementedError("ModalVLLMBackend is planned for Phase 2")

    def get_layer_for_probing(self) -> int:
        raise NotImplementedError("ModalVLLMBackend is planned for Phase 2")
