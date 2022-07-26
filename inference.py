import torch

from models import *
from config import *

WEIGHTS_PATH = ""
IMG_PATH = ""

model = Generator()
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

out = model(IMG_PATH)
