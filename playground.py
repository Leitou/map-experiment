import torch

from custom_types import Attack

if __name__ == "__main__":
    print("hello world")
    print(Attack.DELAY, Attack.DELAY.value)
    print(f'GPU available: {torch.cuda.is_available()}')
