import json
import os
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from jinja2 import Environment, FileSystemLoader

from phantomFacts.inference_methods.backend_inference_interface import LiteLLMInterface
from phantomFacts.utils import extract_and_eval_json


def evaluate_dataset(
    responses_dataset: Dataset,
    evaluators: list[str] = ["phantomFacts_rubric"],
    eval_results_dir: str = "results/eval_results",
    name_suffix: str = "",
    force_reeval: bool = False,
    adjust_scores: bool = True,
) -> pd.DataFrame:
    """
    Evaluate model responses using specified evaluators.

    Args:
        responses_dataset: Dataset containing model responses
        evaluators: List of evaluator names
        eval_results_dir: Directory to save evaluation results
        name_suffix: Optional suffix for output files
        force_reeval: Whether to force re-evaluation even if results exist
        adjust_scores: Whether to adjust only deem response hallucination if
            a novel claim is made.

    Returns:
        DataFrame with evaluation results
    """
    os.makedirs(eval_results_dir, exist_ok=True)

    # Check for existing aggregated results first
    if "phantomFacts_rubric" not in evaluators:
        raise ValueError("Currently only phantomFacts_rubric evaluator is supported")

    # Load evaluation template
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    path_template = os.path.join(path_this_file, "templates")
    env = Environment(loader=FileSystemLoader(path_template))
    eval_template = env.get_template("judge_template.j2")

    # Create evaluator model using LiteLLMInterface
    evaluator_model = LiteLLMInterface(
        model_name="openai/gpt-4o", max_send_messages=50, response_format={"type": "json_object"}
    )

    all_results = []

    # Group by model and system prompt
    for model_name in responses_dataset.unique("model"):
        short_model_name = model_name.split("/")[-1]

        for sys_prompt in responses_dataset.unique("sys_prompt"):
            print(f"Evaluating {short_model_name} with {sys_prompt}")
            # Check for existing results for this model/prompt combination
            output_name = f"{short_model_name}_{sys_prompt}"
            if name_suffix:
                output_name += f"_{name_suffix}"
            output_file = os.path.join(eval_results_dir, f"{output_name}_eval.json")

            if os.path.exists(output_file) and not force_reeval:
                print(f"Loading existing evaluation results for {output_name}")
                with open(output_file, "r") as f:
                    results_for_model = json.load(f)

                # Apply score adjustments if needed
                if adjust_scores:
                    for result in results_for_model:
                        no_relevant_fields = result["raw_evaluation"]["no_relevant_facts_evaluation"]

                        # Adjust scores
                        for field in no_relevant_fields.values():
                            if (
                                field["score"] == 2
                                and "flags" in field
                                and "novel_claims" in field["flags"]
                                and field["flags"]["novel_claims"] == False
                            ):
                                field["score"] = 1

                        # Recalculate hallucination_control
                        proper_abstentions_or_mild_answer = sum(
                            1 for field in no_relevant_fields.values() if field["score"] == 0 or field["score"] == 1
                        )
                        result["hallucination_control"] = proper_abstentions_or_mild_answer / len(no_relevant_fields)

                all_results.extend(results_for_model)
                continue

            # Rest of the evaluation logic remains the same...
            subset = responses_dataset.filter(lambda x: x["model"] == model_name and x["sys_prompt"] == sys_prompt)

            # Generate evaluation prompts
            eval_prompts = []
            for item in subset:
                try:

                    prompt = eval_template.render(
                        given_facts=item["facts"],
                        no_relevant_facts=item["no_relevant_facts"],
                        model_response=item["response"],
                    )
                    eval_prompts.append(prompt)
                except Exception as e:
                    print(f"Error processing response: {str(e)}\nResponse: {item['response'][:100]}...")
                    continue

            if not eval_prompts:
                print(f"No valid prompts for {model_name} with {sys_prompt}")
                continue
            if len(eval_prompts) != len(subset):
                print(f"[WARNING]: Percentage of responses that were parsed: {len(eval_prompts) / len(subset) * 100}%")

            # Convert to conversations and get evaluations
            conversations = [[{"role": "user", "content": prompt}] for prompt in eval_prompts]
            eval_responses = evaluator_model.complete_batch_of_conversations(conversations)

            # Process evaluation results
            results_for_model = []
            for eval_response in eval_responses:
                try:
                    eval_result = extract_and_eval_json(eval_response)[0]
                    no_relevant_fields = eval_result["no_relevant_facts_evaluation"]

                    # Adjust scores if needed (only in memory)
                    if adjust_scores:
                        for field in no_relevant_fields.values():
                            if (
                                field["score"] == 2
                                and "flags" in field
                                and "novel_claims" in field["flags"]
                                and field["flags"]["novel_claims"] == False
                            ):
                                field["score"] = 1

                    proper_abstentions_or_mild_answer = sum(
                        1 for field in no_relevant_fields.values() if field["score"] == 0 or field["score"] == 1
                    )

                    hallucination_control = proper_abstentions_or_mild_answer / len(no_relevant_fields)

                    result = {
                        "model": model_name,
                        "sys_prompt": sys_prompt,
                        "hallucination_control": hallucination_control,
                        "raw_evaluation": eval_result,
                    }
                    results_for_model.append(result)
                    all_results.append(result)

                except Exception as e:
                    print(f"Failed to parse evaluation result for {model_name} with {sys_prompt}: {e}")
                    continue

            if len(results_for_model) != len(eval_responses):
                print(
                    f"[WARNING]: Percentage of evalaution that were successfully parsed: {len(results_for_model) / len(eval_responses) * 100}%"
                )

            # Save results for this model/prompt combination
            if results_for_model:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(results_for_model, f, indent=2)

    if not all_results:
        raise ValueError("No evaluation results were successfully generated")

    return pd.DataFrame(all_results)
