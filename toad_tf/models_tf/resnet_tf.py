import tensorflow as tf

def resnet50_baseline(pretrained=True):
    return tf.keras.applications.ResNet50(
        include_top=False,        
        weights='imagenet' if pretrained else None,
        pooling='avg'             
    )
