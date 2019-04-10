import tensorflow as tf

def transform(voxels, matrix):
    num_batch = voxels.get_shape().as_list()[0]
    depth = voxels.get_shape().as_list()[1]
    height = voxels.get_shape().as_list()[2]
    width = voxels.get_shape().as_list()[3]
    channels = voxels.get_shape().as_list()[4]

    grid_x = tf.reshape(tf.tile(tf.linspace(-1., 1., width), [height * depth]), [depth, height, width])
    grid_y = tf.transpose(tf.reshape(tf.tile(tf.linspace(1., -1., height), [width * depth]), [depth, width, height]), [0, 2, 1])
    grid_z = tf.transpose(tf.reshape(tf.tile(tf.linspace(-0., 1., depth), [width * height]), [height, width, depth]), [2, 0, 1])
    ones = tf.ones_like(grid_x)

    grid = tf.stack([grid_x, grid_y, grid_z, ones], axis=-1)

    grid_t = tf.reshape(grid, [-1, 4]) # flatten to list of coordinates
    grid_t = tf.transpose(grid_t, [1, 0])

    matrix = tf.reshape(matrix, (4, 4))
    matrix = tf.to_float(matrix)
    grid_t = tf.matmul(matrix, grid_t) # apply inverse mvp

    w_t = tf.slice(grid_t, [3, 0], [1, -1])
    grid_t /= w_t # perspective division

    idx = tf.slice(grid_t, [0, 0], [3, -1])
    idx = tf.transpose(idx, [1, 0])
    idx = tf.reshape(idx, [1, depth, height, width, 3])

    x_i = tf.slice(idx, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
    y_i = tf.slice(idx, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1])
    z_i = tf.slice(idx, [0, 0, 0, 0, 2], [-1, -1, -1, -1, 1])

    x_i = tf.reshape(x_i, [-1])
    y_i = tf.reshape(y_i, [-1])
    z_i = tf.reshape(z_i, [-1])

    out = interpolate(voxels, x_i, y_i, z_i, (depth, height, width))
    out = tf.reshape(
        out,
        tf.stack([1, depth, height, width, channels]))
    return out

def interpolate(im, x, y, z, out_size):
  """Bilinear interploation layer.
  Args:
    im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
      It is the input volume for the transformation layer (tf.float32).
    x: A tensor of size [num_batch, out_depth, out_height, out_width]
      representing the inverse coordinate mapping for x (tf.float32).
    y: A tensor of size [num_batch, out_depth, out_height, out_width]
      representing the inverse coordinate mapping for y (tf.float32).
    z: A tensor of size [num_batch, out_depth, out_height, out_width]
      representing the inverse coordinate mapping for z (tf.float32).
    out_size: A tuple representing the output size of transformation layer
      (float).
  Returns:
    A transformed tensor (tf.float32).
  """

  def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
      rep = tf.transpose(
          tf.expand_dims(tf.ones(shape=tf.stack([
              n_repeats,
          ])), 1), [1, 0])
      rep = tf.to_int32(rep)
      x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
      return tf.reshape(x, [-1])

  with tf.variable_scope('_interpolate'):
    num_batch = im.get_shape().as_list()[0]
    depth = im.get_shape().as_list()[1]
    height = im.get_shape().as_list()[2]
    width = im.get_shape().as_list()[3]
    channels = im.get_shape().as_list()[4]

    x = tf.to_float(x)
    y = tf.to_float(y)
    z = tf.to_float(z)
    depth_f = tf.to_float(depth)
    height_f = tf.to_float(height)
    width_f = tf.to_float(width)
    # Number of disparity interpolated.
    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    zero = tf.zeros([], dtype='int32')
    # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
    max_z = tf.to_int32(tf.shape(im)[1] - 1)
    max_y = tf.to_int32(tf.shape(im)[2] - 1)
    max_x = tf.to_int32(tf.shape(im)[3] - 1)

    # Converts scale indices from [-1, 1] to [0, width/height/depth].
    x = (x) * (width_f)
    y = (y) * (height_f)
    z = (z) * (depth_f)

    x0 = tf.to_int32(tf.floor(x))
    x1 = x0 + 1
    y0 = tf.to_int32(tf.floor(y))
    y1 = y0 + 1
    z0 = tf.to_int32(tf.floor(z))
    z1 = z0 + 1

    x0_clip = tf.clip_by_value(x0, zero, max_x)
    x1_clip = tf.clip_by_value(x1, zero, max_x)
    y0_clip = tf.clip_by_value(y0, zero, max_y)
    y1_clip = tf.clip_by_value(y1, zero, max_y)
    z0_clip = tf.clip_by_value(z0, zero, max_z)
    z1_clip = tf.clip_by_value(z1, zero, max_z)
    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth
    base = _repeat(
        tf.range(num_batch) * dim1, out_depth * out_height * out_width)
    base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
    base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
    base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
    base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clip
    idx_z0_y0_x1 = base_z0_y0 + x1_clip
    idx_z0_y1_x0 = base_z0_y1 + x0_clip
    idx_z0_y1_x1 = base_z0_y1 + x1_clip
    idx_z1_y0_x0 = base_z1_y0 + x0_clip
    idx_z1_y0_x1 = base_z1_y0 + x1_clip
    idx_z1_y1_x0 = base_z1_y1 + x0_clip
    idx_z1_y1_x1 = base_z1_y1 + x1_clip

    # Use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
    i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
    i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
    i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
    i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
    i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
    i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
    i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)

    # Finally calculate interpolated values.
    x0_f = tf.to_float(x0)
    x1_f = tf.to_float(x1)
    y0_f = tf.to_float(y0)
    y1_f = tf.to_float(y1)
    z0_f = tf.to_float(z0)
    z1_f = tf.to_float(z1)
    # Check the out-of-boundary case.
    x0_valid = tf.to_float(
        tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
    x1_valid = tf.to_float(
        tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
    y0_valid = tf.to_float(
        tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
    y1_valid = tf.to_float(
        tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
    z0_valid = tf.to_float(
        tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
    z1_valid = tf.to_float(
        tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))

    w_z0_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                1)
    w_z0_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                1)
    w_z0_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                1)
    w_z0_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                1)
    w_z1_y0_x0 = tf.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                1)
    w_z1_y0_x1 = tf.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                1)
    w_z1_y1_x0 = tf.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                1)
    w_z1_y1_x1 = tf.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                1)

    output = tf.add_n([
        w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1,
        w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1,
        w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1,
        w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1
    ])
    return output

