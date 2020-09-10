import numpy as np
from quiver_engine import server
from torchvision import  models
from quiver_engine.model_utils import register_hook

import threading

if __name__ == "__main__":
    model = models.vgg19(pretrained=False)

    hook_list = register_hook(model)

    thread = threading.Thread(target=server.launch, args=(model, hook_list, "./data/Cat",True, [200,200], ))
    thread.start()
    # thread.join() #block untile thread finish

    print("do other things below")
    for i in range(5):
        print(i)
    