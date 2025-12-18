
import sys

print("Checking for Hailo libraries...")

try:
    import hailo_platform
    print("[OK] hailo_platform is installed.")
except ImportError:
    print("[MISSING] hailo_platform is NOT installed.")

try:
    import hailo
    print("[OK] 'hailo' module is installed.")
except ImportError:
    print("[MISSING] 'hailo' module is NOT installed.")

print("\nPython executable:", sys.executable)
