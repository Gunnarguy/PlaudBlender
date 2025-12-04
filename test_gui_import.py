
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import gui.app...")
    import gui.app
    print("Successfully imported gui.app")

    print("Attempting to import gui.main_window...")
    import gui.main_window
    print("Successfully imported gui.main_window")

    print("Attempting to import gui.tabs.dashboard...")
    import gui.tabs.dashboard
    print("Successfully imported gui.tabs.dashboard")

    print("All imports successful.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
