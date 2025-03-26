import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

from datasets import Dataset
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from phantomFacts.inference_methods.answer_w_context import AnswerWithContext
from phantomFacts.inference_methods.backend_inference_interface import BaseLLMInterface, LiteLLMInterface, VLLMInterface


def load_template(
    template_name: str, template_path: str = "/workspace/src/phantomFacts/data/make_synth_data/recipe_templates"
) -> Any:
    """Load a Jinja2 template from the specified path."""
    env = Environment(loader=FileSystemLoader(template_path))
    return env.get_template(template_name)


def create_model_instance(model_config: Dict[str, str]) -> BaseLLMInterface:
    """Create a model instance based on the configuration."""
    # Extract model params with defaults
    model_params = model_config.get("model_params", {})

    # Set max_send_messages based on model provider
    if model_config["inference_backend"] == "litellm":
        # Default to 50 for OpenAI models, 20 for others
        # anthropic is slow
        default_max_messages = (
            50 if "openai" in model_config["model"] else 15 if "anthropic" in model_config["model"] else 25
        )
        max_send_messages = model_params.get("max_send_messages", default_max_messages)
        # remove max_send_messages from model_params
        model_params.pop("max_send_messages", None)
        # Set default temperature if not provided
        if "temperature" not in model_params:
            model_params["temperature"] = 1.0

        return LiteLLMInterface(model_name=model_config["model"], max_send_messages=max_send_messages, **model_params)
    elif model_config["inference_backend"] == "vllm":

        sampling_params = (
            {"temperature": 1} if "sampling_params" not in model_params else model_params["sampling_params"]
        )

        return VLLMInterface(model_path=model_config["model"], sampling_params=sampling_params)
    else:
        raise ValueError(f"Unsupported inference backend: {model_config['inference_backend']}")


def generate_to_dataset(
    dataset: Dataset,
    models: list[dict[str, str]],
    generation_modes: dict[str, list[dict[str, str]]] | list[list[dict[str, str]]] | None,
    response_output_dir: str = "results/responses",
    name_suffix: str = "",
    force_regen: bool = False,
    use_multiple_templates: bool = True,
    generation_seed: int = 42,
) -> Dataset:
    """
    Generate responses for each model-generation mode combination.

    Args:
        dataset: Input dataset
        models: List of model configurations
        generation_modes: List of generation modes, each containing:
            - mode_name: Name of the generation mode
            - insert_before_model_instructions: Messages to insert before user query
        response_output_dir: Directory to save responses
        template_name: Name of the template file to use
        name_suffix: Optional suffix for output files
        force_regen: Whether to regenerate responses even if they exist
        template_names: List of template names to use. If supplied, should be the same lenght as datasets
        **kwargs: Additional arguments for the inference method
    """
    os.makedirs(response_output_dir, exist_ok=True)
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    path_template = os.path.join(path_this_file, "templates", "fill_in_templates")
    template_names = os.listdir(path_template)
    random.seed(generation_seed)
    if use_multiple_templates:
        template_names = random.choices(template_names, k=len(dataset))
        print("INFO: distribution of templates:", dict(Counter(template_names)))
        templates = [load_template(template_name, path_template) for template_name in template_names]
    else:
        template_names = ["fill_in_template_default.j2"] * len(dataset)
        templates = [load_template(template_name, path_template) for template_name in template_names]

    all_responses = []

    for model_config in tqdm(models, desc="Processing models", leave=True):
        model_name = model_config["model"].split("/")[-1]

        # Check if all mode responses exist before creating model instance
        all_modes_exist = True
        for mode in generation_modes:
            output_name = f"{model_name}_{mode['mode_name']}"
            if name_suffix:
                output_name += f"_{name_suffix}"
            output_file = os.path.join(response_output_dir, f"{output_name}.json")

            if not os.path.exists(output_file) or force_regen:
                all_modes_exist = False
                break

        # If all responses exist, load them and continue to next model
        if all_modes_exist and not force_regen:
            for mode in generation_modes:
                output_name = f"{model_name}_{mode['mode_name']}"
                if name_suffix:
                    output_name += f"_{name_suffix}"
                output_file = os.path.join(response_output_dir, f"{output_name}.json")
                print(f"Loading existing responses from {output_file}")
                with open(output_file, "r") as f:
                    responses_data = json.load(f)
                    all_responses.extend(responses_data)
            continue

        # Create model instance only if needed
        model = create_model_instance(model_config)
        for mode in tqdm(generation_modes, desc=f"Processing modes for {model_name}", leave=True):
            print(f"Generating responses for {model_name} and mode {mode['mode_name']}")
            output_name = f"{model_name}_{mode['mode_name']}"
            if name_suffix:
                output_name += f"_{name_suffix}"
            output_file = os.path.join(response_output_dir, f"{output_name}.json")

            # Check if responses exist (in case of partial completion)
            if os.path.exists(output_file) and not force_regen:
                print(f"Loading existing responses from {output_file}")
                with open(output_file, "r") as f:
                    responses_data = json.load(f)
                    all_responses.extend(responses_data)
                continue

            # Generate new responses
            inference_method = AnswerWithContext(llm=model)

            prompts = [
                template.render(document_type=item["document_type"], fields=item["fields"], facts=item["facts"])
                for item, template in zip(dataset, templates)
            ]

            conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
            responses, metadata = inference_method.answer(
                conversations, mode=mode, progress_bar=dict(desc=f"Generating responses", leave=True, flush=True)
            )

            # Create response data
            responses_data = []
            for i, (response, meta) in enumerate(zip(responses, metadata)):
                response_data = {
                    **dataset[i],
                    "model": model_config["model"],
                    "sys_prompt": mode["mode_name"],
                    "response": response,
                    "metadata": meta,
                }
                responses_data.append(response_data)
                all_responses.append(response_data)

            # Save responses
            with open(output_file, "w") as f:
                json.dump(responses_data, f, indent=2)
        model.destroy()

    return Dataset.from_list(all_responses)
