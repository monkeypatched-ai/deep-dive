def TurnOnRobot():
    """ turns on the robot and updates the llm """
    print("Power On")
    return "Perform System Checks"

def RunDiagnosticTests():
    """ performs system checks """
    return "Initialize Sensors"

def InitializeSensors(parameters):
    """ initialize the sensors"""
    return "Initialize Actuators"

def InitializeActuators(parameters):
    """ initialize the actuators """
    return "Set Initial Configuration"

def SetInitialConfiguration(parameters):
    """ sets the initial configuration """
    return "Establish Communication"

def InitializeGraph():
    """ initalize the graph """
    return "Device inititalized.."