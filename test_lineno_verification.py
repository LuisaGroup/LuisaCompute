#!/usr/bin/env python3
"""
Test to verify correct line number reporting.
This script should report an error on line 5 (the 'return a' line).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'py'))

from luisa import *

@func
def f():
    return a

# This should trigger an error pointing to line 14 (return a)
try:
    init()
    f(dispatch_size=1)
except Exception as e:
    print("Exception caught (expected)")
    import traceback
    traceback.print_exc()
