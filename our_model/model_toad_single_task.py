import tensorflow as tf

# -------------------------------
#   Attention Network (Gated)
# -------------------------------

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(tf.keras.layers.Layer):

    def __init__(self, D = 256, dropout = False):
        
        
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [tf.keras.layers.Dense(D, activation="tanh")]
        
        self.attention_b = [tf.keras.layers.Dense(D, activation="sigmoid")]
        
        if dropout:
            self.attention_a.append(tf.keras.layers.Dropout(0.25))
            self.attention_b.append(tf.keras.layers.Dropout(0.25))

        self.attention_a = tf.keras.Sequential(self.attention_a)
        self.attention_b = tf.keras.Sequential(self.attention_b)
        
        self.attention_c = tf.keras.layers.Dense(1)
    
    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a*b
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class TOAD_fc_single_task(tf.keras.Model):
    def __init__(self, size_arg="big", dropout=False, n_classes=2):
        super().__init__()

        size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = size_dict[size_arg]

        fc = [tf.keras.layers.Dense(size[1], activation="relu")]
        if dropout:
            fc.append(tf.keras.layers.Dropout(0.25))
        fc.append(tf.keras.layers.Dense(size[1], activation="relu"))
        if dropout:
            fc.append(tf.keras.layers.Dropout(0.25))
        
        attention_net = Attn_Net_Gated(D = size[2], dropout = dropout)
        
        self.fc_layers = tf.keras.Sequential(fc)
        self.attention_net = attention_net
        self.classifier = tf.keras.layers.Dense(n_classes)

    def call(self, h, return_features=False, attention_only=False):

        h = self.fc_layers(h)
        A, h = self.attention_net(h)  
        A = tf.transpose(A, perm=[1,0])
        if attention_only:
            return A[0]
        
        A = tf.nn.softmax(A, axis=1) 
        M = tf.linalg.matmul(A, h)

        logits  = self.classifier(M) 
        Y_hat = tf.math.top_k(logits, k=1).indices
        Y_hat = tf.squeeze(Y_hat)
        Y_prob = tf.nn.softmax(logits)

        out = {
            "logits": logits,
            "Y_prob": Y_prob,
            "Y_hat": tf.squeeze(Y_hat),
            "A": A
        }

        if return_features:
            out["features"] = M

        return out
