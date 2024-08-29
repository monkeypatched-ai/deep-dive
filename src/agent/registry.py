from  tests.agent.tasks import TurnOnRobot,RunDiagnosticTests,InitializeSensors,InitializeActuators,SetInitialConfiguration,InitializeGraph

function_registry = {
    "TurnOnRobot":TurnOnRobot,
    "RunDiagnosticTests":RunDiagnosticTests,
    "InitializeSensors":InitializeSensors,
    "InitializeActuators":InitializeActuators,
    "SetInitialConfiguration":SetInitialConfiguration,
    "InitializeGraph":InitializeGraph
}
