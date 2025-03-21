#!/usr/bin/env python
"""
Custom test runner that applies TensorFlow and matplotlib patches before running tests
"""
import os
import sys

# Apply TensorFlow environment variables before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU memory growth
os.environ['TF_PLUGGABLE_DEVICE_LIBRARY_PATH'] = ''  # Disable pluggable device loading

# Configure matplotlib to use non-interactive backend
os.environ['MPLBACKEND'] = 'Agg'  # Force non-interactive backend

# Set flag to indicate we're in test mode
os.environ['ENHANCED_PM_TESTING'] = 'true'

# Mock modules that might cause issues
MOCK_MODULES = ['matplotlib.pyplot', 'seaborn']

class MockModule:
    """Mock module to replace problematic imports"""
    
    def __init__(self, name):
        self.name = name
        self._mocked_functions = {}
        
    def __getattr__(self, attr):
        if attr not in self._mocked_functions:
            # Create a mock function that does nothing and returns None
            self._mocked_functions[attr] = lambda *args, **kwargs: None
        return self._mocked_functions[attr]

# Monkey patch TensorFlow modules that cause crashes
def apply_compatibility_patches():
    """Apply patches to prevent crashes during testing"""
    # Apply inspect patches for Python 3.9+
    import inspect
    import functools

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
        
    # Patch import system to handle problematic modules
    import sys
    from types import ModuleType
    
    # Intercept module loading
    original_import = __import__
    
    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Check if this is a module we want to mock
        if name in MOCK_MODULES:
            print(f"Mocking module: {name}")
            return MockModule(name)
            
        # For matplotlib submodules that might cause issues
        if name.startswith('matplotlib.') and name != 'matplotlib.use':
            try:
                return original_import(name, globals, locals, fromlist, level)
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Error importing {name}: {str(e)}. Using mock instead.")
                return MockModule(name)
            
        # Handle TensorFlow imports specially
        if name == 'tensorflow' or name.startswith('tensorflow.'):
            try:
                module = original_import(name, globals, locals, fromlist, level)
                
                # If importing tensorflow, apply patches
                if hasattr(module, 'config') and hasattr(module.config, 'experimental'):
                    # Force CPU only mode
                    try:
                        list_physical_devices_original = module.config.experimental.list_physical_devices
                        
                        def safe_list_physical_devices(device_type):
                            if device_type.lower() == 'gpu':
                                return []  # Return empty list for GPUs
                            return list_physical_devices_original(device_type)
                        
                        module.config.experimental.list_physical_devices = safe_list_physical_devices
                    except Exception:
                        pass
                        
                return module
            except Exception as e:
                print(f"Error importing {name}: {str(e)}. Using mock instead.")
                return MockModule(name)
                
        # For all other imports, proceed normally
        try:
            return original_import(name, globals, locals, fromlist, level)
        except ImportError as e:
            # If the import fails and it's matplotlib related, mock it
            if 'matplotlib' in name:
                print(f"Error importing {name}: {str(e)}. Using mock instead.")
                return MockModule(name)
            # Re-raise the exception for non-matplotlib modules
            raise
    
    # Apply the patched import
    sys.__import__ = patched_import
    sys.modules['builtins'].__import__ = patched_import
    
    # Pre-mock modules to prevent any import attempts
    for module_name in MOCK_MODULES:
        sys.modules[module_name] = MockModule(module_name)
        
    # Handle matplotlib specially - force Agg backend
    try:
        import matplotlib
        matplotlib.use('Agg')  # Force non-interactive backend
    except ImportError:
        sys.modules['matplotlib'] = MockModule('matplotlib')
        sys.modules['matplotlib.pyplot'] = MockModule('matplotlib.pyplot')

# Apply patches before importing pytest
apply_compatibility_patches()

print("Applied compatibility patches for TensorFlow and matplotlib")

# Now import and run pytest
import pytest

if __name__ == "__main__":
    # Get command line arguments
    args = sys.argv[1:]
    
    print("Running tests with compatibility patches applied...")
    
    # Use default test directory if no args provided
    if not args:
        args = ['tests']
    
    # Add verbosity flag if not specified
    if not any(arg.startswith('-v') for arg in args):
        args.append('-v')
    
    # Run pytest with the arguments
    sys.exit(pytest.main(args))
