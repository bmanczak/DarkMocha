from typing import Any, Dict, List

from phantomFacts.inference_methods.backend_inference_interface import BaseLLMInterface


class AnswerWithContext:
    """
    Implementation that answers queries with optional context messages inserted
    before the user query.
    """

    def __init__(self, llm: BaseLLMInterface):
        """Initialize with an LLM interface."""
        self.llm = llm

    def answer(
        self, conversations: list[list[dict[str, str]]], mode: Dict[str, Any] | None = None, **kwargs
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        Process conversations with optional context messages.

        Args:
            conversations: List of conversations to process
            mode: Generation mode containing:
                - mode_name: Name of the generation mode
                - insert_before_model_instructions: Messages to insert before user query
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (list of answers, list of metadata)
        """
        # Handle single conversation case
        is_single = isinstance(conversations[0], dict)
        conversations = [conversations] if is_single else conversations

        # Create working copy of conversations
        processed_conversations = []
        for conv in conversations:
            # Start with context messages if provided
            new_conv = []
            if mode and mode.get("insert_before_model_instructions"):
                # Filter out empty system messages for Anthropic models
                instructions = mode["insert_before_model_instructions"]
                if hasattr(self.llm, "model_name") and "anthropic" in self.llm.model_name.lower():
                    instructions = [
                        msg for msg in instructions if not (msg["role"] == "system" and msg["content"].strip() == "")
                    ]
                new_conv.extend(instructions)
            # Add the original conversation messages
            new_conv.extend(conv)
            processed_conversations.append(new_conv)

        # Get responses from LLM
        answers = self.llm.complete_batch_of_conversations(processed_conversations)

        # Create metadata
        metadata = [
            {
                "mode_name": mode["mode_name"] if mode else None,
                "system_prompt": next((msg["content"] for msg in conv if msg["role"] == "system"), None),
            }
            for conv in processed_conversations
        ]

        if is_single:
            return answers[0], metadata[0]
        return answers, metadata
