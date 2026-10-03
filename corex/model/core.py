
import torch
from torch import nn

from .representation import RepresentationEngine
from .state import PersistentState
from .heads import PredictionHead
from corex.memory.working import WorkingMemory
from corex.reasoning.core import ReasoningCore


class COREXModel(nn.Module):
    """
    Integrated CORE-X V0.1 prototype.

    Pipeline:
        Input
          -> Representation
          -> Persistent State
          -> Working Memory Read
          -> Iterative Reasoning
          -> Prediction
          -> Working Memory Write
    """

    def __init__(
        self,
        vocab_size=64,
        d_model=256,
        d_state=256,
        working_memory_slots=16,
        reasoning_steps=4,
        num_classes=2,
    ):
        super().__init__()

        self.representation = RepresentationEngine(
            vocab_size=vocab_size,
            d_model=d_model,
        )

        self.persistent_state = PersistentState(
            d_model=d_model,
            d_state=d_state,
        )

        self.working_memory = WorkingMemory(
            slots=working_memory_slots,
            d_model=d_state,
        )

        self.reasoning = ReasoningCore(
            d_state=d_state,
            reasoning_steps=reasoning_steps,
        )

        self.prediction_head = PredictionHead(
            d_state=d_state,
            num_classes=num_classes,
        )

    def forward(
        self,
        input_ids,
        state=None,
        memory=None,
    ):
        # 1. Represent input
        representations = self.representation(input_ids)

        # 2. Update persistent global state
        if state is None:
            state = self.persistent_state.initial_state(
                batch_size=input_ids.size(0),
                device=input_ids.device,
            )

        state = self.persistent_state(
            representations,
            state,
        )

        # 3. Read working memory
        if memory is None:
            memory = self.working_memory.initial_memory(
                batch_size=input_ids.size(0),
                device=input_ids.device,
            )

        memory_context = self.working_memory.read(
            memory,
            state,
        )

        # 4. Iterative reasoning
        final_state, reasoning_states = self.reasoning(
            state,
            memory_context,
        )

        # 5. Write refined information into working memory
        updated_memory = self.working_memory.write(
            memory,
            final_state,
        )

        # 6. Prediction
        logits = self.prediction_head(final_state)

        return {
            "representations": representations,
            "state": state,
            "memory_context": memory_context,
            "final_state": final_state,
            "memory": updated_memory,
            "reasoning_states": reasoning_states,
            "logits": logits,
        }
