import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

class CNNEncoder(keras.Model):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        
        # Extract model type from model_name
        if 'resnet50' in model_name.lower() or model_name == 'resnet50_trunc':
            self.model_type = 'resnet50'
        elif 'efficientnet' in model_name.lower() or model_name == 'efficientnet':
            self.model_type = 'efficientnet'
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported: resnet50, efficientnet")
        
        self.model_name = model_name
        self.pool = pool
        
        # Build the base model
        if self.model_type == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(None, None, 3)
            )

            self.base_model = keras.Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('conv4_block6_out').output
            )
        elif self.model_type == 'efficientnet':
            self.base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(None, None, 3)
            )
            # EfficientNetB0 outputs 1280 features, project to 1024 to match ResNet50
            self.projection_layer = Dense(1024, use_bias=True)
        else:
            self.projection_layer = None
        
        # Add pooling layer if needed
        if pool:
            self.pool_layer = GlobalAveragePooling2D()
        else:
            self.pool_layer = None
    
    def call(self, x, training=False):
        
        out = self.base_model(x, training=training)
        
        if self.pool_layer:
            out = self.pool_layer(out)
        
        # Apply projection for EfficientNet to match ResNet50 feature dimension (1024)
        if self.projection_layer is not None:
            out = self.projection_layer(out)
        
        return out