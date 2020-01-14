---
layout: post
title: "Nicer Causal Convs for Tensorflow 2"
tags: [conv1d, causal, wavenet, autoregressive, tensorflow, dilated]
comments: true
share: true
mathjax: false
excerpt_separator: "<!--more-->"
---


Causal 1D convolution are quite useful when working with autoregressive models like WaveNet. To define this layer in tensorflow 2 we just pass "causal" padding in Conv1D layer arguments as is the following one liner: 

```python
conv_layer = tf.keras.layers.Conv1D(
    filters=64, 
    kernel_size=2,
    dilation_rate=4, 
    padding='causal')
```

Calling this layer will preserve the temporal dimension of the input by adding left padding which is not always desirable.  As we stack more and more layers with larger dilation rate padding will become a large portion of the input data. Also the default implementation of dilated convolution layer implicitly define some padding based on your first input restricting your choice of the input shape later on.


<!--more-->


```python
x = np.random.randn(1, 150, 96).astype(np.float32)
conv_layer(x).shape
--> TensorShape([1, 150, 64])
```

Now if you try to feed some other data with different shape you'll get an error:

```python
y = np.random.randn(1, 140, 96).astype(np.float32)
conv_layer(y)
--> InvalidArgumentError: padded_shape[0]=146 is not divisible by block_shape[0]=4 [Op:SpaceToBatchND]
```

We can handle dilation ourselves as we want in DilatedConv1D class that inherits from the built-in Conv1D layer.


```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def time_to_batch(value, dilation, name=None):
    shape = tf.shape(value)
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
    reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])
    return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]]), pad_elements


def batch_to_time(value, dilation, name=None):
    shape = tf.shape(value)
    prepared = tf.reshape(value, [dilation, -1, shape[2]])
    transposed = tf.transpose(prepared, perm=[1, 0, 2])
    return tf.reshape(transposed,
                      [tf.divide(shape[0], dilation), -1, shape[2]])


class DilatedConv1D(tf.keras.layers.Conv1D):
    
    def __init__(self, nb_filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(nb_filters, kernel_size, dilation_rate=1, **kwargs)
        self.custom_dilation = dilation_rate if isinstance(dilation_rate, tuple) else (dilation_rate, )
    
    def call(self, input_tensor, **kwargs):
        if self.custom_dilation [0] > 1:
            x, pad = time_to_batch(input_tensor, self.custom_dilation[0])
            x = super().call(x)
            output = batch_to_time(x, self.custom_dilation[0])
            width = tf.shape(output)[1] - pad
            output = output[:,:width]
        else:
            output = super().call(input_tensor)
        return output
```

Now we can implement CausalConv1D on top of DilatedConv1D so that we will be able to handle varying input shapes and control whether we want causal left padding or not with TF-like padding keywords: "valid" and "same".


```python
class CausalConv1D(DilatedConv1D):
    
    def __init__(self, nb_filters, kernel_size, padding='valid', **kwargs):
        allowed_paddings = ['same', 'valid']
        if padding not in allowed_paddings:
            raise ValueError('Unknown padding, allowed: %s' % str(allowed_paddings))
        super().__init__(nb_filters, kernel_size, **kwargs)
        self.causal_padding = padding

    def call(self, input_tensor,  **kwargs):
        pad = self.custom_dilation[0] * (self.kernel_size[0] - 1)
        if self.causal_padding == 'same':
            causal_padding = [[0, 0], [pad, 0], [0, 0]]
            padded_tensor = tf.pad(input_tensor, paddings=causal_padding)
            output = super().call(padded_tensor)
        else: 
            out_width = tf.shape(input_tensor)[1] - pad
            output = super().call(input_tensor)[:, :out_width]
            
        return output
    
```

Let's try our new causal layer in action. We'll create two identical architectures but with different causal layer implementations and since we set the same weights for both networks they should return the same result.


```python
class OriginalNetwork(tf.keras.Model):
    def __init__(self):
        super(OriginalNetwork, self).__init__()

        self.layer_stack = [
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='causal', activation='relu', dilation_rate=1),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='causal', activation='relu', dilation_rate=2),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding='causal', activation='relu', dilation_rate=4),
            tf.keras.layers.Conv1D(1, kernel_size=3, padding='causal', dilation_rate=8)
        ]

    def call(self, input_tensor, **kwargs):
        x = input_tensor
        for layer in self.layer_stack:
            x = layer(x, **kwargs)
        return x

    
class ModifiedNetwork(tf.keras.Model):
    def __init__(self, padding='valid'):
        super(ModifiedNetwork, self).__init__()

        self.layer_stack = [
            CausalConv1D(32, kernel_size=3, activation='relu', padding=padding, dilation_rate=1),
            CausalConv1D(32, kernel_size=3, activation='relu', padding=padding, dilation_rate=2),
            CausalConv1D(32, kernel_size=3, activation='relu', padding=padding, dilation_rate=4),
            CausalConv1D(1, kernel_size=3, padding=padding, dilation_rate=8)
        ]
            

    def call(self, input_tensor, training=True, **kwargs):
        x = input_tensor
        for layer in self.layer_stack:
            x = layer(x, **kwargs)
        return x
    
```


```python
x = np.random.randn(1, 150, 96).astype(np.float32)
net1 = OriginalNetwork()
net2 = ModifiedNetwork(padding='same')
_, _ = net1(x), net2(x)
net1.set_weights(net2.get_weights())
output1, output2 = net1(x), net2(x)
print('Shapes are equal: ' + str((output1.shape == output2.shape)))
print('Values are equal: ' + str((output1 == output2).numpy().all()))
```

Outputs:

    Shapes are equal: True
    Values are equal: True

With valid padding the modified network should output reduced temporal dimension.

```python
x = np.random.randn(1, 150, 96).astype(np.float32)
net1 = OriginalNetwork()
net2 = ModifiedNetwork(padding='valid')
_, _ = net1(x), net2(x)
net1.set_weights(net2.get_weights())
output1, output2 = net1(x), net2(x)
print('Shapes are: ' + str((output1.shape, output2.shape)))
print('Values are equal: ' + str((output1[0,30:] == output2[0]).numpy().all()))

```

Outputs:

    Shapes are: (TensorShape([1, 150, 1]), TensorShape([1, 120, 1]))
    Values are equal: True




