
import numpy as np
from quiver_engine import server
from quiver_engine.model_utils import register_hook
from torchvision import  models

if __name__ == "__main__":
    model = models.resnet18(pretrained=False)

    hook_list = register_hook(model)
    
    server.launch(model, hook_list, input_folder="./data/Cat", image_size=[250,250], use_gpu=True)

