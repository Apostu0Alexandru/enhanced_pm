"""
Matplotlib compatibility patch for headless environments

This patch configures matplotlib to use a non-interactive backend
and patches problematic imports to avoid C extension issues.
"""

import os
import sys
import importlib.util
import logging

logger = logging.getLogger(__name__)

def apply_matplotlib_patch():
    """Apply patches for matplotlib to work in headless environments"""
    logger.info("Applying matplotlib patch for headless environment")
    
    # Set a non-interactive backend before importing matplotlib
    os.environ['MPLBACKEND'] = 'Agg'  # Use the Agg backend (non-interactive)
    
    # Disable OpenMP for MacOS to avoid thread issues
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # Inform user that patch was applied
    logger.info("Applied matplotlib patch - using Agg backend")
    print("Applied matplotlib patch - using Agg backend")

# Apply the patch when imported
apply_matplotlib_patch()
