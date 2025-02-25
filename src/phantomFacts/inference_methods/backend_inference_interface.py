import time
from typing import Any, Dict, List, Optional, Tuple

from litellm import batch_completion
from tqdm import tqdm

# will be init in vllm class if needed
LLM = None
SamplingParams = None
AutoTokenizer = None


class BaseLLMInterface:
    """
    Minimal interface describing two methods: complete_conversation() and
    complete_batch_of_conversations().

    We define them specifically for multi-turn conversation use:
      - complete_conversation(conversation: List[dict], **kwargs) -> str
      - complete_batch_of_conversations(list_of_conversations: List[List[dict]], **kwargs) -> List[str]

    Each conversation is a list of message dicts:
      [ {"role": "system"/"user"/"assistant", "content": "..."},
        ... ]

    The model is expected to produce a single assistant completion for each conversation.
    """

    def complete_conversation(self, conversation: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    def complete_batch_of_conversations(
        self, batch_of_conversations: List[List[Dict[str, str]]], **kwargs
    ) -> List[str]:
        raise NotImplementedError

    def destroy(self):
        pass


###############################################################################
# Dynainference-based LiteLLM Interface
###############################################################################


class LiteLLMInterface(BaseLLMInterface):
    """
    An implementation of BaseLLMInterface using litellm's batch_completion.

    Args:
        model_name (str): The model identifier (e.g. "gpt-3.5-turbo", "anthropic/claude-3-opus-20240229")
        max_send_messages (int): Maximum number of messages to send in a single batch
        **model_params: Additional parameters to pass to the model (temperature, etc)
    """

    def __init__(self, model_name: str, max_send_messages: int = 20, **model_params):
        """Initialize LiteLLM interface.

        Args:
            model_name: Name of the model to use
            max_send_messages: Maximum number of messages to send in one request
            **model_params: Additional parameters to pass to the model (temperature, etc)
        """
        super().__init__()
        print(f"Initializing LiteLLMInterface with model_name: {model_name}")
        self.model_name = model_name
        self.max_send_messages = max_send_messages
        self.model_params = model_params

    def complete_conversation(self, conversation: List[Dict[str, str]], **kwargs) -> str:
        """Complete a single conversation."""
        outputs = self.complete_batch_of_conversations([conversation], **kwargs)
        return outputs[0] if outputs else ""

    def complete_batch_of_conversations(
        self, batch_of_conversations: List[List[Dict[str, str]]], **kwargs
    ) -> List[str]:
        """Complete multiple conversations in batches."""
        generations = []
        max_retries = 3  # Maximum number of retries for failed batches

        for i in tqdm(
            range(0, len(batch_of_conversations), self.max_send_messages),
            desc=f"Processing batched requests (max bs={self.max_send_messages})",
            dynamic_ncols=True,
            position=0,
            leave=True,
        ):
            batch = batch_of_conversations[i : i + self.max_send_messages]
            retry_count = 0
            batch_success = False

            while not batch_success and retry_count < max_retries:
                try:
                    responses = batch_completion(model=self.model_name, messages=batch, **self.model_params, **kwargs)
                    # Validate each response before extending
                    valid_responses = []
                    for response in responses:
                        try:
                            content = response["choices"][0]["message"]["content"]
                            valid_responses.append(response)
                        except (KeyError, IndexError, TypeError):
                            raise ValueError(f"Invalid response structure: {response}")

                    # If we get here, all responses were valid
                    generations.extend(valid_responses)
                    batch_success = True

                except Exception as e:
                    print(f"Batch failed (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2**retry_count)  # Exponential backoff
                    else:
                        print(f"Batch failed after {max_retries} attempts, skipping...")
                        # Add empty responses for the failed batch
                        generations.extend([{"choices": [{"message": {"content": ""}}]} for _ in batch])

            # Wait to respect rate limits
            time.sleep(1)

        # Extract text from responses
        return [response["choices"][0]["message"]["content"] for response in generations]

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """Generate a response using LiteLLM."""
        try:
            # Pass model parameters along with the request
            response = batch_completion(
                model=self.model_name, messages=messages[-self.max_send_messages :], **self.model_params
            )
            return response.choices[0].message.content, {"finish_reason": response.choices[0].finish_reason}
        except Exception as e:
            print(f"Error generating response: {e}")
            return "", {"error": str(e)}


###############################################################################
# Dynainference-based VLLM Interface
###############################################################################


class VLLMInterface(BaseLLMInterface):
    """
    An implementation of BaseLLMInterface using vllm's LLM.

    Args:
        model_path (str): The model path/name (e.g. "meta-llama/Llama-2-7b-chat-hf")
        default_sampling (Optional[Dict[str, Any]]): Default sampling parameters
    """

    def __init__(self, model_path: str, sampling_params: Optional[Dict[str, Any]] = None, **kwargs):

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        if LLM is None or SamplingParams is None or AutoTokenizer is None:
            raise EnvironmentError("vllm or its dependencies are not installed or not importable.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        sampling_params = SamplingParams(**(sampling_params or {}))
        self.llm = LLM(model=model_path, max_model_len=8192, **kwargs)
        self.sampling_params = sampling_params

    def _prepare_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Apply the model's chat template to the conversation."""
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    def complete_conversation(self, conversation: List[Dict[str, str]], **kwargs) -> str:
        """Complete a single conversation."""
        prompt = self._prepare_prompt(conversation)
        outputs = self.llm.generate(prompts=[prompt], sampling_params=self.sampling_params, **kwargs)
        return outputs[0].outputs[0].text if outputs else ""

    def complete_batch_of_conversations(
        self, batch_of_conversations: List[List[Dict[str, str]]], **kwargs
    ) -> List[str]:
        """Complete multiple conversations in parallel."""
        prompts = [self._prepare_prompt(conv) for conv in batch_of_conversations]
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params, **kwargs)
        return [output.outputs[0].text for output in outputs]

    def destroy(self):
        import gc

        import torch

        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
