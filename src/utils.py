"""
utils.py
Utility functions for exporting generated test cases
to CSV and JSON formats.
"""

import csv
import json
import io
import re
from datetime import datetime


def export_to_csv(test_cases_text: str) -> str:
    """
    Parse plain-text test cases and export to CSV format.

    Args:
        test_cases_text: Raw string output from the LLM

    Returns:
        str: CSV content as a string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["TC ID", "Title", "Preconditions", "Steps", "Expected Result", "Priority", "Type"])

    # Simple parser: split by TC- pattern
    lines = test_cases_text.strip().split("\n")
    current_tc = {}
    current_field = None
    steps_buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect TC ID line: TC-001: Title
        tc_match = re.match(r"(TC-\d+):\s*(.*)", line)
        if tc_match:
            # Save previous TC
            if current_tc.get("id"):
                current_tc["steps"] = " | ".join(steps_buffer)
                writer.writerow([
                    current_tc.get("id", ""),
                    current_tc.get("title", ""),
                    current_tc.get("preconditions", ""),
                    current_tc.get("steps", ""),
                    current_tc.get("expected", ""),
                    current_tc.get("priority", "Medium"),
                    current_tc.get("type", "Positive")
                ])
            current_tc = {"id": tc_match.group(1), "title": tc_match.group(2)}
            steps_buffer = []
            current_field = None
            continue

        if line.lower().startswith("precondition"):
            current_field = "preconditions"
            current_tc["preconditions"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("expected result"):
            current_field = "expected"
            current_tc["expected"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("priority"):
            current_tc["priority"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("type"):
            current_tc["type"] = line.split(":", 1)[-1].strip()
        elif re.match(r"^\d+\.", line):
            steps_buffer.append(line)
            current_field = "steps"
        elif current_field == "steps":
            steps_buffer.append(line)
        elif current_field and current_field != "steps":
            current_tc[current_field] = current_tc.get(current_field, "") + " " + line

    # Write last TC
    if current_tc.get("id"):
        current_tc["steps"] = " | ".join(steps_buffer)
        writer.writerow([
            current_tc.get("id", ""),
            current_tc.get("title", ""),
            current_tc.get("preconditions", ""),
            current_tc.get("steps", ""),
            current_tc.get("expected", ""),
            current_tc.get("priority", "Medium"),
            current_tc.get("type", "Positive")
        ])

    return output.getvalue()


def export_to_json(test_cases_text: str, original_input: str) -> str:
    """
    Export generated test cases to JSON format.

    Args:
        test_cases_text: Raw string output from the LLM
        original_input: The original user input

    Returns:
        str: JSON content as a string
    """
    payload = {
        "generated_at": datetime.now().isoformat(),
        "original_input": original_input,
        "raw_output": test_cases_text,
        "test_cases": _parse_to_list(test_cases_text)
    }
    return json.dumps(payload, indent=2)


def _parse_to_list(text: str) -> list:
    """Helper: parse LLM output into a list of dicts."""
    test_cases = []
    lines = text.strip().split("\n")
    current = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        tc_match = re.match(r"(TC-\d+):\s*(.*)", line)
        if tc_match:
            if current.get("id"):
                test_cases.append(current)
            current = {"id": tc_match.group(1), "title": tc_match.group(2), "details": []}
        elif current:
            current.setdefault("details", []).append(line)

    if current.get("id"):
        test_cases.append(current)

    return test_cases
