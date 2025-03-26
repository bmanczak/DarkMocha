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

                    def get_value(field, key):
                        """
                        Safely retrieve the value for 'key' from 'field'.
                        It looks at the top level, then inside 'evidence',
                        while gracefully handling unexpected types or structures.
                        """
                        if not isinstance(field, dict):
                            # If 'field' isn't a dictionary (e.g., it's an int, string, None, etc.), return None
                            return None

                        # 1) Check if 'key' is directly in 'field'
                        if key in field:
                            return field[key]

                        # 2) Check if there's an 'evidence' sub-dict and if 'key' is in there
                        evidence = field.get("evidence")
                        if isinstance(evidence, dict) and key in evidence:
                            return evidence[key]

                        # If not found, return None
                        return None

                    def set_value(field, key, value):
                        """
                        Safely set the value for 'key' in 'field'.
                        It sets at the top level if 'key' is found there,
                        otherwise inside 'evidence' if that's a dict containing 'key'.
                        No-op if 'field' or 'evidence' is not a dict.
                        """
                        if not isinstance(field, dict):
                            return  # Can't set a key on non-dict types

                        # 1) If 'key' is already at the top level, set it there
                        if key in field:
                            field[key] = value
                            return

                        # 2) Otherwise, if 'evidence' is a dict and has 'key', set it there
                        evidence = field.get("evidence")
                        if isinstance(evidence, dict) and key in evidence:
                            evidence[key] = value

                    for ix, result in enumerate(results_for_model):
                        if result["hallucination_control"] is None:
                            print(
                                f"WARNING: Had to skip {ix}th result for {model_name} with {sys_prompt} because hallucination control was None. Here's the raw evaluation: {result['raw_evaluation']}"
                            )
                            continue
                        # Check if raw_evaluation exists and is a dictionary
                        if not isinstance(result.get("raw_evaluation"), dict):
                            print(
                                f"WARNING: Invalid raw evaluation format for {model_name} with {sys_prompt} at index {ix}."
                            )
                            print("Cannot adjust scores for this result.")
                            continue

                        # Check if no_relevant_facts_evaluation exists
                        if "no_relevant_facts_evaluation" not in result["raw_evaluation"]:
                            print(
                                f"WARNING: No relevant facts evaluation found for {model_name} with {sys_prompt} at index {ix}."
                            )
                            print("Cannot adjust scores for this result.")
                            continue

                        no_relevant_fields = result["raw_evaluation"]["no_relevant_facts_evaluation"]

                        # Adjust scores in a generalized, safe way
                        for field_key, field in no_relevant_fields.items():
                            score = get_value(field, "score")
                            flags = get_value(field, "flags")

                            # Check for your existing logic: if score == 2 and novel_claims == False => set score = 1
                            if score == 2 and flags is not None and flags.get("novel_claims") is False:
                                set_value(field, "score", 1)

                        # Recalculate hallucination_control
                        proper_abstentions_or_mild_answer = sum(
                            1 for field in no_relevant_fields.values() if get_value(field, "score") in (0, 1)
                        )
                        if len(no_relevant_fields) == 0:
                            result["hallucination_control"] = 0  # or some fallback
                        else:
                            result["hallucination_control"] = proper_abstentions_or_mild_answer / len(
                                no_relevant_fields
                            )

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
            num_failed_eval_parsing = 0
            for eval_response in eval_responses:
                eval_result = eval_response
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
                    result = {
                        "model": model_name,
                        "sys_prompt": sys_prompt,
                        "hallucination_control": None,
                        "raw_evaluation": eval_result,
                    }
                    results_for_model.append(result)
                    all_results.append(result)
                    num_failed_eval_parsing += 1
                    continue

            # Always print parsing statistics, not just when there are failures
            print(
                f"Evaluation parsing statistics for {model_name} with {sys_prompt}: "
                f"Successfully parsed {(len(results_for_model) - num_failed_eval_parsing) / len(eval_responses) * 100:.2f}% "
                f"({len(results_for_model) - num_failed_eval_parsing}/{len(eval_responses)})"
            )

            # Save results for this model/prompt combination
            if results_for_model:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(results_for_model, f, indent=2)

    if not all_results:
        raise ValueError("No evaluation results were successfully generated")

    return pd.DataFrame(all_results)


def evaluate_over_refusal_dataset(
    responses_dataset: Dataset,
    eval_results_dir: str = "results/eval_results",
    name_suffix: str = "",
    force_reeval: bool = False,
) -> pd.DataFrame:
    """
    Evaluate the model responses for over-refusal and detail hallucionation on the fields that should be filled.
    Only evaluates fields with directly relevant facts, excluding interpolatable fields.

    Args:
        responses_dataset: Dataset containing model responses
        eval_results_dir: Directory to save evaluation results
        name_suffix: Optional suffix for output files
        force_reeval: Whether to force re-evaluation even if results exist

    Returns:
        DataFrame with evaluation results, filtered to include only directly relevant facts
    """
    os.makedirs(eval_results_dir, exist_ok=True)

    # Extract experiment name from eval_results_dir
    experiment_name = os.path.basename(eval_results_dir)
    responses_dir = os.path.join(os.path.dirname(os.path.dirname(eval_results_dir)), "responses", experiment_name)

    # Load evaluation template
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    path_template = os.path.join(path_this_file, "templates")
    env = Environment(loader=FileSystemLoader(path_template))
    eval_template = env.get_template("judge_over_refusal_template.j2")

    # Create evaluator model using LiteLLMInterface
    evaluator_model = LiteLLMInterface(
        model_name="openai/gpt-4o", max_send_messages=50, response_format={"type": "json_object"}
    )

    all_results = []

    # Load response files (similar to refusal_judge_validation.py approach)
    print(f"Looking for response files in {responses_dir}")
    responses = {}
    if os.path.exists(responses_dir):
        print("Loading model response files...")
        for filename in os.listdir(responses_dir):
            if filename.endswith(".json"):
                with open(os.path.join(responses_dir, filename), "r") as f:
                    try:
                        response_data = json.load(f)
                        responses[filename] = response_data
                    except json.JSONDecodeError:
                        print(f"Error loading response file: {filename}")
        print(f"Loaded {len(responses)} response files")
    else:
        print(f"Warning: Response directory {responses_dir} not found. Filtering may be less effective.")

    # Group by model and system prompt
    for model_name in responses_dataset.unique("model"):
        short_model_name = model_name.split("/")[-1]

        for sys_prompt in responses_dataset.unique("sys_prompt"):
            print(f"Evaluating {short_model_name} with {sys_prompt}")
            # Check for existing results for this model/prompt combination
            output_name = f"{short_model_name}_{sys_prompt}_over_refusal"
            if name_suffix:
                output_name += f"_{name_suffix}"
            output_file = os.path.join(eval_results_dir, f"{output_name}_eval.json")

            if os.path.exists(output_file) and not force_reeval:
                print(f"Loading existing evaluation results for {output_name}")
                with open(output_file, "r") as f:
                    results_for_model = json.load(f)

                # Alert about using existing results vs new filtering logic
                print("NOTE: Using cached results. Applying filtering to exclude interpolatable fields.")

                # Find corresponding response file for this model/prompt
                response_filename = f"{short_model_name}_{sys_prompt}.json"
                if response_filename in responses:
                    print(f"Found matching response file: {response_filename}")
                    response_data_list = responses[response_filename]
                else:
                    # Try alternative filenames with name_suffix
                    if name_suffix:
                        alt_response_filename = f"{short_model_name}_{sys_prompt}_{name_suffix}.json"
                        if alt_response_filename in responses:
                            print(f"Found matching response file: {alt_response_filename}")
                            response_data_list = responses[alt_response_filename]
                        else:
                            print(f"Warning: No matching response file found for {output_name}")
                            # Try to proceed with limited filtering capabilities
                            response_data_list = None
                    else:
                        print(f"Warning: No matching response file found for {output_name}")
                        response_data_list = None

                filtered_results = []
                filtered_count = 0

                # Process existing results to filter out interpolatable fields from metrics
                for idx, result in enumerate(results_for_model):
                    # Keep the original result intact
                    processed_result = result.copy()

                    # Skip if raw_evaluation is missing/invalid or has no relevant facts section
                    if (
                        not isinstance(result.get("raw_evaluation"), dict)
                        or "relevant_facts_evaluation" not in result["raw_evaluation"]
                    ):
                        filtered_results.append(processed_result)
                        continue

                    # Get relevant facts evaluation from the raw evaluation
                    relevant_fields_eval = result["raw_evaluation"]["relevant_facts_evaluation"]

                    # If we have response data, use it to identify interpolatable fields
                    if response_data_list and idx < len(response_data_list):
                        response_item = response_data_list[idx]

                        # Get interpolatable fields from the response data
                        interpolatable_fields = response_item.get("interpolatable_facts", [])

                        # Filter fields to exclude interpolatable ones
                        filtered_fields = {
                            field: eval_data
                            for field, eval_data in relevant_fields_eval.items()
                            if field not in interpolatable_fields
                        }

                        # Skip if no fields left after filtering
                        if not filtered_fields:
                            filtered_results.append(processed_result)
                            continue

                        # Safely extract scores from fields, handling both dict and non-dict values
                        refusal_count = 0
                        hallucination_count = 0
                        valid_field_count = 0

                        for field_value in filtered_fields.values():
                            if isinstance(field_value, dict) and "score" in field_value:
                                valid_field_count += 1
                                if field_value["score"] == 2:
                                    refusal_count += 1
                                elif field_value["score"] == 1:
                                    hallucination_count += 1
                            elif isinstance(field_value, int):
                                # Handle case where field value is directly the score
                                valid_field_count += 1
                                if field_value == 2:
                                    refusal_count += 1
                                elif field_value == 1:
                                    hallucination_count += 1

                        # Only recalculate metrics if we have valid fields
                        if valid_field_count > 0:
                            # Recalculate metrics based on filtered fields
                            over_refusal = refusal_count / valid_field_count
                            detail_hallucination = hallucination_count / valid_field_count

                            # Update metrics in the processed result
                            processed_result["over_refusal"] = over_refusal
                            processed_result["detail_hallucionation"] = detail_hallucination

                            # Add fields for debugging/verification
                            processed_result["filtered_fields_count"] = valid_field_count
                            processed_result["total_fields_count"] = len(relevant_fields_eval)
                            processed_result["interpolatable_fields_count"] = len(interpolatable_fields)

                            # Add document metadata from response
                            if "domain" not in processed_result and "domain" in response_item:
                                processed_result["domain"] = response_item.get("domain", "")
                            if "document_type" not in processed_result and "document_type" in response_item:
                                processed_result["document_type"] = response_item.get("document_type", "")

                            filtered_count += 1

                    filtered_results.append(processed_result)

                # Report on filtering results
                if filtered_count > 0:
                    print(
                        f"Successfully filtered {filtered_count}/{len(results_for_model)} results to exclude interpolatable fields"
                    )
                else:
                    print("Warning: Could not filter any results - original metrics retained")
                    print("Consider using force_reeval=True to regenerate results with proper filtering")

                all_results.extend(filtered_results)
                continue

            # Rest of the evaluation logic (generating new evaluations)...
            subset = responses_dataset.filter(lambda x: x["model"] == model_name and x["sys_prompt"] == sys_prompt)

            # Generate evaluation prompts
            eval_prompts = []
            subset_items = []  # Keep track of items to retrieve document metadata later
            for item in subset:
                # Modified to exclude interpolatable fields - only use directly relevant facts
                if "directly_relevant_facts" in item:
                    # If directly_relevant_facts is available, use it directly
                    relevant_fields = item["directly_relevant_facts"]
                else:
                    # Otherwise, filter out both no_relevant_facts and interpolatable_facts
                    interpolatable_fields = item.get("interpolatable_facts", [])
                    relevant_fields = [
                        field
                        for field in item["fields"]
                        if field not in item["no_relevant_facts"] and field not in interpolatable_fields
                    ]

                if not relevant_fields:
                    # Skip items with no directly relevant fields
                    continue

                try:
                    prompt = eval_template.render(
                        given_facts=item["facts"],
                        relevant_fields=relevant_fields,
                        model_response=item["response"],
                    )
                    eval_prompts.append(prompt)
                    subset_items.append(item)  # Store the item for later metadata retrieval
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
            num_failed_eval_parsing = 0
            for idx, eval_response in enumerate(eval_responses):
                eval_result = eval_response
                try:
                    eval_result = extract_and_eval_json(eval_response)[0]
                    relevant_fields = eval_result["relevant_facts_evaluation"]

                    # Safely calculate metrics
                    refusal_count = 0
                    hallucination_count = 0
                    valid_field_count = 0

                    for field_value in relevant_fields.values():
                        if isinstance(field_value, dict) and "score" in field_value:
                            valid_field_count += 1
                            if field_value["score"] == 2:
                                refusal_count += 1
                            elif field_value["score"] == 1:
                                hallucination_count += 1
                        elif isinstance(field_value, int):
                            valid_field_count += 1
                            if field_value == 2:
                                refusal_count += 1
                            elif field_value == 1:
                                hallucination_count += 1

                    if valid_field_count == 0:
                        print(f"Warning: No valid field scores found for evaluation at index {idx}")
                        over_refusal_percent = 0
                        detail_hallucionation_percent = 0
                    else:
                        over_refusal_percent = refusal_count / valid_field_count
                        detail_hallucionation_percent = hallucination_count / valid_field_count

                    # Include domain and document_type in the result for easier filtering later
                    item = subset_items[idx]
                    result = {
                        "model": model_name,
                        "sys_prompt": sys_prompt,
                        "domain": item.get("domain", ""),
                        "document_type": item.get("document_type", ""),
                        "id": item.get("id", ""),
                        "over_refusal": over_refusal_percent,
                        "detail_hallucionation": detail_hallucionation_percent,
                        "raw_evaluation": eval_result,
                        # Add fields for verification
                        "filtered_fields_count": valid_field_count,
                        "original_fields_count": len(item.get("fields", [])),
                        "interpolatable_fields_count": len(item.get("interpolatable_facts", [])),
                    }
                    results_for_model.append(result)
                    all_results.append(result)

                except Exception as e:
                    print(f"Failed to parse evaluation result for {model_name} with {sys_prompt}: {e}")
                    result = {
                        "model": model_name,
                        "sys_prompt": sys_prompt,
                        "over_refusal": None,
                        "detail_hallucionation": None,
                        "raw_evaluation": eval_response,  # Use the raw response for debugging
                    }
                    results_for_model.append(result)
                    all_results.append(result)
                    num_failed_eval_parsing += 1
                    continue

            # Always print parsing statistics, not just when there are failures
            print(
                f"Evaluation parsing statistics for {model_name} with {sys_prompt}: "
                f"Successfully parsed {(len(results_for_model) - num_failed_eval_parsing) / len(eval_responses) * 100:.2f}% "
                f"({len(results_for_model) - num_failed_eval_parsing}/{len(eval_responses)})"
            )

            # Save results for this model/prompt combination
            if results_for_model:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as f:
                    json.dump(results_for_model, f, indent=2)

    if not all_results:
        raise ValueError("No over-refusal evaluation results were successfully generated")

    return pd.DataFrame(all_results)
