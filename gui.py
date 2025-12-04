#!/usr/bin/env python3
"""
PlaudBlender GUI - Refactored Entry Point
"""
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.app import main

if __name__ == "__main__":
    main()
