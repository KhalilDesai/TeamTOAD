import os
from functools import partial
from timm_wrapper import CNNEncoder
from constants import MODEL2CONSTANTS
from transform_utils import get_eval_transforms
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    model = CNNEncoder(model_name=model_name)
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms