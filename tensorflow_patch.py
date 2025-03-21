"""
Patch for TensorFlow compatibility with Python 3.9+ 
This adds the missing inspect.ArgSpec functionality
and disables problematic hardware acceleration features
"""
import inspect
import functools
import os

# Disable GPU and hardware acceleration to prevent crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU memory growth

# Disable TensorFlow pluggable device operations that might cause crashes
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = ''  # Disable pluggable device loading

# Create a replacement for the removed ArgSpec
if not hasattr(inspect, 'ArgSpec'):
    # Create a namedtuple similar to the old ArgSpec
    inspect.ArgSpec = functools.namedtuple(
        'ArgSpec', 
        ['args', 'varargs', 'keywords', 'defaults']
    )
    
    # Add a getargspec function similar to the original
    def getargspec(func):
        sig = inspect.signature(func)
        args = []
        defaults = []
        varargs = None
        keywords = None
        
        for param_name, param in sig.parameters.items():
            kind = param.kind
            
            if kind == inspect.Parameter.POSITIONAL_ONLY or kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                args.append(param_name)
                if param.default != inspect.Parameter.empty:
                    defaults.append(param.default)
            elif kind == inspect.Parameter.VAR_POSITIONAL:
                varargs = param_name
            elif kind == inspect.Parameter.VAR_KEYWORD:
                keywords = param_name
                
        # Match the behavior of the original getargspec
        if not defaults:
            defaults = None
            
        return inspect.ArgSpec(args, varargs, keywords, defaults)
    
    # Add the getargspec function to inspect module
    inspect.getargspec = getargspec

print("Applied TensorFlow patch for Python 3.9+ compatibility - CPU-only mode enabled")
