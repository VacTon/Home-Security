
from hailo_platform import HEF, VDevice
import os

model_path = "models/arcface_mobilefacenet.hef"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
else:
    try:
        hef = HEF(model_path)
        print("HEF object methods:")
        print(dir(hef))
        
        print("\nChecking for create_configure_params...")
        if hasattr(hef, 'create_configure_params'):
            print("Found create_configure_params")
        else:
            print("NOT found create_configure_params")
            
        vdevice = VDevice()
        print("\nVDevice object methods:")
        print(dir(vdevice))
        
    except Exception as e:
        print(f"Error: {e}")
