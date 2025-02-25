import json
import re
from typing import Dict, List, Optional


class SanitizeDecoder(json.JSONDecoder):
    """
    A custom JSON decoder that sanitizes the input string before decoding.

    This decoder applies specific regex replacements to handle common JSON formatting issues,
    such as unescaped backslashes and trailing commas in arrays or objects.

    Attributes:
        None

    Methods:
        decode(s, **kwargs): Sanitizes and decodes the input JSON string.

    Usage:
        decoder = SanitizeDecoder()
        parsed_json = decoder.decode(json_string)
    """

    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r"([^\\])\\([^\\])"), r"\1\\\\\2"),
            (re.compile(r",(\s*])"), r"\1"),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)


def extract_and_eval_json(input_text: str, max_jsons: Optional[int] = 1) -> List[Dict]:
    """
    Extracts JSON-like substrings from a given input text and converts them into Python dictionaries.
    This function tries to handle cases where JSON objects might not be properly closed by assuming
    they end at the end of the string. It stops parsing once the specified number of JSON objects is reached.

    Parameters:
    - input_text (str): A string that potentially contains one or more JSON-like substrings.
    - max_jsons (Optional[int]): The maximum number of JSON objects to parse. Defaults to 1. If None, all JSON objects will be parsed.

    Returns:
    - list: A list of dictionaries. Each dictionary is a successfully parsed JSON object from
            the input text. If no valid JSON objects are found, or if all JSON-like substrings
            fail to parse, the list will be empty.

    Raises:
    - json.JSONDecodeError: If `json.loads()` encounters a string that is not valid JSON, this
                            exception will be caught and handled internally by the function.
                            The function will continue to parse other substrings, if any.

    Example Usage:
    --------------
    example_string = '''
    {
    "system_check_result": "The response could lead to harm.",
    "route": "potential_violation",
    "system_tip": "Decline the offer and seek help from trusted institutions."
    '''
    parsed_jsons = extract_and_eval_json(example_string, max_jsons=1)
    for parsed_json in parsed_jsons:
        print(parsed_json)

    Output:
    -------
    [{'system_check_result': 'The response could lead to harm.', 'route': 'potential_violation',
    'system_tip': 'Decline the offer and seek help from trusted institutions.'}]
    """
    # Common JSON pitfalls
    input_text = input_text.replace("True", "true")
    input_text = input_text.replace("False", "false")
    input_text = input_text.replace("None", "null")
    input_text = input_text.replace("none", "null")
    input_text = input_text.replace("\n", "")

    results = []
    if max_jsons is None:
        max_jsons = float("inf")

    # First, try parsing the entire input as a single JSON object
    try:
        parsed_dict = json.loads(input_text)
        return [parsed_dict]
    except json.JSONDecodeError:
        # If standard parsing fails, proceed with the original method
        pass

    # If the above fails, use the original method with SanitizeDecoder
    brace_count = 0
    start_index = None
    json_count = 0

    for index, char in enumerate(input_text):
        if char == "{":
            if brace_count == 0:
                start_index = index
            brace_count += 1
        elif char == "}":
            brace_count -= 1

        if brace_count == 0 and start_index is not None:
            json_candidate = input_text[start_index : index + 1]
            try:
                # First try standard json.loads
                parsed_dict = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    # If standard fails, try SanitizeDecoder
                    parsed_dict = json.loads(json_candidate, cls=SanitizeDecoder)
                except json.JSONDecodeError:
                    continue

            results.append(parsed_dict)
            json_count += 1
            if json_count >= max_jsons:
                return results
            start_index = None

    # Attempt to fix and parse if we have an unclosed JSON object at the end
    if brace_count != 0 and start_index is not None:
        json_candidates: list[str] = [
            input_text[start_index:] + "}",  # unclosed json
            input_text[start_index:] + '..."' + "}",  # unclosed json without ending quoa
            input_text[start_index:] + "...'" + "}",  # unclosed json with ending quoa
        ]
        for json_candidate in json_candidates:
            try:
                parsed_dict = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed_dict = json.loads(json_candidate, cls=SanitizeDecoder)
                except json.JSONDecodeError:
                    pass
            else:  # Code that runs if NO exception occurs in the try block
                results.append(parsed_dict)
                break  # break out of the loop if we found a valid json

    return results
