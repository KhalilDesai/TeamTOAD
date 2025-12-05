import tensorflow as tf

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
        
        self.attention_c = tf.keras.layers.Dense(D)
    
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

class TOAD_fc_mtl_concat(tf.keras.layers.Layer):
    def __init__(self, size_arg = "big", dropout = False, n_classes = 2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [tf.keras.layers.Dense(size[1], activation="relu")]
        if dropout:
            fc.append(tf.keras.layers.Dropout(0.25))
        fc.append(tf.keras.layers.Dense(size[1], activation="relu"))
        if dropout:
            fc.append(tf.keras.layers.Dropout(0.25))
        
        attention_net = Attn_Net_Gated(D = size[2], dropout = dropout)
        
        fc.append(attention_net)
        self.attention_net = tf.keras.Sequential(fc)
        self.classifier = tf.keras.layers.Dense(n_classes)
        self.site_classifier = tf.keras.layers.Dense(2)
                    
    def call(self, h, sex, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = tf.transpose(A, perm=[1,0])
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = tf.nn.softmax(A, axis=1) 
        M = tf.linalg.matmul(A, h)
        sex = tf.reshape(sex, (1, 1))
        M = tf.concat([M, sex], axis=1)

        logits  = self.classifier(M) 
        Y_hat = tf.math.top_k(logits, k=1).indices
        Y_hat = tf.squeeze(Y_hat)
        Y_prob = tf.nn.softmax(logits)

        site_logits  = self.site_classifier(M) 
        site_hat = tf.math.top_k(site_logits, k=1).indices
        site_hat = tf.squeeze(site_hat)
        site_prob = tf.nn.softmax(site_logits)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 
                'site_logits': site_logits, 'site_prob': site_prob, 'site_hat': site_hat, 'A': A_raw})

        return results_dict

