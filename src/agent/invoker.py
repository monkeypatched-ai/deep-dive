from registry import function_registry

def invoke(function_name, *args, **kwargs):
    if function_name in function_registry:
        return function_registry[function_name](*args, **kwargs)
    else:
        raise ValueError(f"Function '{function_name}' is not registered.")
