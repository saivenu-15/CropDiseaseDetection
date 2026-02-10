
import torch
import numpy as np

print(f"Torch version: {torch.__version__}")
print(f"Numpy version: {np.__version__}")

try:
    a = np.array([1, 2, 3])
    t = torch.from_numpy(a)
    print("Success: torch + numpy working")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

try:
    import cv2
    print(f"Opencv version: {cv2.__version__}")
    print("Success: opencv import working")
except ImportError as e:
    print(f"Error importing cv2: {e}")
except Exception as e:
    print(f"Error with cv2: {e}")
