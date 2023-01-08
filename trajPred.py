
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Layer
import tensorflow.keras.layers as nn
from tensorflow.keras import Sequential, Model

from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()

        self.norm = nn.LayerNormalization()
        self.fn = fn

    def call(self, x, context=None, training=True):
        if context is not None:
            return self.fn(self.norm(x), context, training=training)
        else:
            return self.fn(self.norm(x), training=training)

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()

        def GELU():
            def gelu(x, approximate=False):
                if approximate:
                    coeff = tf.cast(0.044715, x.dtype)
                    return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
                else:
                    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

            return nn.Activation(gelu)

        self.net = Sequential([
            nn.Dense(units=hidden_dim),
            GELU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_q = nn.Dense(units=inner_dim, use_bias=False)
        self.to_kv = nn.Dense(units=inner_dim * 2, use_bias=False)

        if project_out:
            self.to_out = [
                nn.Dense(units=dim),
                nn.Dropout(rate=dropout)
            ]
        else:
            self.to_out = []

        self.to_out = Sequential(self.to_out)

    def call(self, x, context, training=True):
        q = self.to_q(x)
        kv = self.to_kv(context)
        kv = tf.split(kv, num_or_size_splits=2, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, *(kv)))

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x

class Transformer(Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(MLP(dim, mlp_dim, dropout=dropout))
            ])

    def call(self, x, context, training=True):
        for attn, mlp in self.layers:
            x = attn(x, context=context, training=training) + x
            x = mlp(x, training=training) + x

        return x

class TrajPred(tf.keras.Model):
    """
    Trajectory prediction module
    """
    def __init__(self):
        super(TrajPred, self).__init__()

        
        dim = 384

        # embedding
        emb_dropout = 0.1

        # transformer
        depth = 6
        heads = 6
        dim_head = 64
        mlp_dim = 384
        dropout = 0.1

        # out
        num_classes = 25 # 8 * 3 + 1

        image_height, image_width = (256, 384)
        patch_height, patch_width = (16, 16)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # patching and positional embedding
        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_height, p2=patch_width),
            nn.Dense(units=dim)
        ], name='patch_embedding')
        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]))
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]))
        self.dropout = nn.Dropout(rate=emb_dropout)

        # latents for multimodality
        self.latents = tf.Variable(initial_value=tf.random.normal([1, 1, 6, dim]))
        self.modes_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=dim)
        ], name='modes_head')

        # transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final head
        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ], name='mlp_head')

        dummy_scene = tf.random.uniform([1,256,384])
        dummy_trajs = tf.random.uniform([1,64,384])
        preds, probs = self(dummy_scene, dummy_trajs)
        self.summary()

    def call(self, scene, trajs, training=True):
        # scene: (B, 256, 384)
        # trajs: (B, 64, 384)
        # out: (B, 64, 6, 8, 2)
        # and (B, 64, 6)

        # patching and positional embedding of the scene
        scene = self.patch_embedding(scene) # (B, 384, 384)
        B, N, D = scene.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        scene = tf.concat([cls_tokens, scene], axis=1) # (B, 385, 384)
        scene += self.pos_embedding[:, :(N + 1)]
        scene = self.dropout(scene, training=training)

        # concat latents to encoded trajectories
        B, N, D = trajs.shape
        x = tf.repeat(trajs[:, :, tf.newaxis, :], repeats=6, axis=2) # (B, 64, 6, 384)
        latents = tf.tile(self.latents, [B, N, 1, 1]) # (B, 64, 6, 384)

        x = tf.concat([x, latents], axis=3) # (B, 64, 6, 768)
        x = self.modes_head(x) # (B, 64, 6, 384)
        x = tf.reshape(x, [-1, 64 * 6, 384]) # (B, 384, 384)

        # apply transformer
        # use encoded trajectories as query and the scene as keys/values
        x = self.transformer(x, scene, training=training) # (B, 384, 384)

        # final layer
        x = self.mlp_head(x) # (B, 384, 25)

        # reshape into trajectory predictions and associated probabilities
        x = tf.reshape(x, [-1, N, 6, 25]) # (B, 64, 6, 25)

        preds = x[:,:,:,:24] # (B, 64, 6, 24)
        preds = tf.reshape(preds, [-1, 64, 6, 8, 3]) # (B, 64, 6, 8, 3)

        probs = x[:,:,:,16] # (B, 64, 6)

        probs = tf.nn.softmax(probs) # as probabilities

        probs = tf.math.log(probs) # as log probabilities

        return preds, probs

if __name__ == "__main__":
    trajPred = TrajPred()