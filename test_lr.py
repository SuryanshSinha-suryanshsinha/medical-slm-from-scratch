from train.train import TrainingConfig, get_lr
from model.model import ModelConfig

cfg = TrainingConfig(model_config=ModelConfig())
print('step 0:    ', get_lr(0, cfg))
print('step 500:  ', get_lr(500, cfg))
print('step 1000: ', get_lr(1000, cfg))
print('step 5500: ', get_lr(5500, cfg))
print('step 10000:', get_lr(10000, cfg))