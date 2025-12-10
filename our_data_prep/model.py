from timm_wrapper import TimmCNNEncoder
from constants import MODEL2CONSTANTS
from transform_utils import get_eval_transforms

# map your logical names to actual timm models + kwargs
TIMM_CONFIGS = {
    "resnet50_trunc": {
        "timm_name": "resnet50.tv_in1k",
        "kwargs": {
            "features_only": True,
            "out_indices": (3,),   # stage giving you 1024-d after GAP
            "pretrained": True,
            "num_classes": 0,
        },
    },
    "efficientnet_b0_trunc": {
        "timm_name": "tf_efficientnet_b0_ns",  # or "efficientnet_b0", etc.
        "kwargs": {
            "features_only": True,
            "out_indices": (4,),  # you can tweak this if you want a different stage
            "pretrained": True,
            "num_classes": 0,
        },
    },
    # you can add uni_v1 / conch_* here too if they use timm backbones
}

def get_encoder(model_name, target_img_size=224):
    constants = MODEL2CONSTANTS[model_name]

    if model_name not in TIMM_CONFIGS:
        raise ValueError(f"Unknown timm model for: {model_name}")

    timm_name = TIMM_CONFIGS[model_name]["timm_name"]
    timm_kwargs = TIMM_CONFIGS[model_name]["kwargs"]

    print(f"loading model checkpoint: {timm_name}")
    model = TimmCNNEncoder(model_name=timm_name, kwargs=timm_kwargs)
    print(model)

    img_transforms = get_eval_transforms(
        mean=constants["mean"],
        std=constants["std"],
        target_img_size=target_img_size,
    )

    return model, img_transforms
