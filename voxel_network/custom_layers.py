import tensorflow as tf
from transform import transform

class ElementwiseMaxPool(tf.keras.layers.Layer):

    def __init__(self, keepdims=False, **kwargs):
        super(ElementwiseMaxPool, self).__init__(**kwargs)
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        return tf.keras.backend.max(inputs, axis=-2, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()[:-2]
        if self.keepdims:
            output_shape += (1,)
        output_shape += (input_shape.as_list()[-1],)
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = {'keepdims': self.keepdims}
        base_config = super(ElementwiseMaxPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReduceSum(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReduceSum, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        out = input_shape.as_list()[:self.axis] + input_shape.as_list()[self.axis + 1:]
        return tf.TensorShape(out)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ReduceSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RepeatElements(tf.keras.layers.Layer):

    def __init__(self, rep, axis=-1, **kwargs):
        super(RepeatElements, self).__init__(**kwargs)
        self.rep = rep
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.keras.backend.repeat_elements(inputs, self.rep, self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] *= self.rep
        return tf.TensorShape(output_shape)

    def get_config(self):
        config = {'rep': self.rep, 'axis': self.axis}
        base_config = super(RepeatElements, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConcatDepth(tf.keras.layers.Layer):
    def __init__(self, axis=-1, num_layers=1, **kwargs):
        super(ConcatDepth, self).__init__(**kwargs)
        self.axis = axis
        self.num_layers = num_layers

    def call(self, inputs, **kwargs):
        batch_size = tf.keras.backend.shape(inputs)[0]
        width = tf.keras.backend.shape(inputs)[3]
        height = tf.keras.backend.shape(inputs)[2]
        depth = tf.keras.backend.shape(inputs)[1]
        dst = tf.range(depth)
        dst = tf.expand_dims(dst, -1)
        dst = tf.expand_dims(dst, -1)
        dst = tf.expand_dims(tf.tile(dst, [1, height, width]), -1)
        dst /= depth
        dst = tf.expand_dims(dst, 0)
        dst = tf.tile(dst, tf.stack((batch_size, 1, 1, 1, self.num_layers)))
        dst = tf.to_float(dst)

        return tf.concat([inputs, dst], axis=self.axis)

    def compute_output_shape(self, input_shape):
        out = input_shape.as_list()
        assert len(out) == 5
        out[self.axis] += self.num_layers
        return tf.TensorShape(out)

    def get_config(self):
        config = {'axis': self.axis, 'num_layers': self.num_layers}
        base_config = super(ConcatDepth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, reduce_dim=False, **kwargs):
        super(ProjectionLayer, self).__init__(**kwargs)
        self.reduce_dim = reduce_dim

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        tensor, proj_matrix = inputs
        tensor = transform(tensor,
                           proj_matrix)
        if self.reduce_dim:
            # reduce depth dimension
            tensor = tf.reduce_max(tensor, axis=1)
        return tensor

    def compute_output_shape(self, input_shape):
        tensor_shape, matrix_shape = input_shape
        assert len(tensor_shape) == 5
        if self.reduce_dim:
            out = tf.TensorShape((tensor_shape[0].value, tensor_shape[2].value, tensor_shape[3].value, tensor_shape[4].value))
        else:
            out = tf.TensorShape((tensor_shape[0].value, tensor_shape[1].value, tensor_shape[2].value, tensor_shape[3].value, tensor_shape[4].value))
        return out

    def get_config(self):
        config = {'reduce_dim': self.reduce_dim}
        base_config = super(ProjectionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))