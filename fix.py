
import torch
import pickle
import io
from pathlib import PosixPath

class PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'WindowsPath':
            return PosixPath
        return super().find_class(module, name)

with open('best.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())
    checkpoint = PathFixUnpickler(buffer).load()

torch.save(checkpoint, 'best_fixed.pt')
print("✅ Fixed model saved to best_fixed.pt")