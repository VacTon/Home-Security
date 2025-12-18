
from hailo_platform import VDevice
import numpy as np

model_path = "models/arcface_mobilefacenet.hef"
vdevice = VDevice()
infer_model = vdevice.create_infer_model(model_path)

print(f"Input name: {infer_model.input().name}")
print(f"Input shape: {infer_model.input().shape}")
print(f"Input dtype: {infer_model.input().dtype}")

print(f"\nOutput name: {infer_model.output().name}")
print(f"Output shape: {infer_model.output().shape}")
print(f"Output dtype: {infer_model.output().dtype}")
