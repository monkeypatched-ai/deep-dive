import re
import json

def parse(input_text):

    # Regex to capture step number, description, and action with optional parameters
    pattern = re.compile(r"Step (\d+): ([\w\s-]+)\nAction: (\w+)(?: \(parameters: ({.*?})\))?")

    # Extracting data
    steps = []
    for match in pattern.finditer(input_text):
        step_number = int(match.group(1))
        description = match.group(2).strip()
        action = match.group(3).strip()
        parameters = json.loads(match.group(4).strip()) if match.group(4) else None

        step = {
            "step": step_number,
            "description": description,
            "action": action
        }
        if parameters:
            step["parameters"] = parameters

        steps.append(step)

    return steps