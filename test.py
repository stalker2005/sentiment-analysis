import torch
print(torch.cuda.is_available())  # Should print True if CUDA is working
print(torch.cuda.current_device())  # Show the current CUDA device
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Show the GPU name
