from src.agent.invoker import invoke
from src.agent.parser import parse

# Input initialization process
# will need to pre train the teacher model to generate the below completion 
# this out put should then be gretrained to the student model
# in the end we should get this as llm output
CONTEXT = "imagine that you are a robot navigating a warehouse before doing this you need to initialise yourself." \
 "in order to do so you will follow the below steps."

PROMPT = "initalize robot"

COMPLETION = """
Here's the initialization process:

Step 1: Power On
Action: TurnOnRobot

Step 2: Perform System Checks
Action: RunDiagnosticTests

Step 3: Initialize Sensors
Action: InitializeSensors (parameters: {"camera":"camera"})

Step 4: Initialize Actuators
Action: InitializeActuators (parameters: {"motors":"motors"})

Step 5: Set Initial Configuration
Action: SetInitialConfiguration (parameters: {"config":"config"})

Step 6: Initialize the map as a graph
Action: InitializeGraph

Now, I'm fully initialized and ready to assist you!

"""

steps = parse(COMPLETION)

for step in steps:
    if "parameters" in step.keys():
        # call the function with the given parameters
        arguments = step["parameters"]
        function_name = step["action"]
        result = invoke(function_name, *arguments)
    else:
        # call without parameters
        function_name = step["action"]
        result = invoke(function_name)

    print(result)
