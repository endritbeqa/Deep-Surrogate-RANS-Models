import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np



#TODO check in general if this is correct
def get_data_from_tfds( config, mode):

    mode = mode+'[:10%]'
    builder = tfds.builder_from_directory(builder_dir=config.data_dir)
    ds = builder.as_dataset(
        split=mode,
        shuffle_files=True
    )

    ds = ds.batch(batch_size=config.batch_size,
                  drop_remainder=False,
                  num_parallel_calls=tf.data.AUTOTUNE
                  )

    return ds.prefetch(tf.data.AUTOTUNE)


def preprocess_data(batch) :

        #TODO always double check this because datasets encode the mask differently
        inputs, targets, label = batch['encoder'], batch['decoder'], batch['label']
        inputs, targets = np.transpose(inputs, (0,3, 1, 2)), np.transpose(targets, (0,3,1,2))

        boundary = inputs[:, 2]
        boundary[boundary != 0] = 1
        b_bound, h_bound, w_bound = boundary.shape
        boundary = np.reshape(boundary, (b_bound,h_bound*w_bound)).astype(bool)
        b, c, h, w = targets.shape

        targets = targets.reshape((b, c, h * w))

        for i in range(b):
            p_mean_i = np.mean(targets[i,0][boundary[i]])
            targets[:, 0][boundary] -= p_mean_i


        max_targets_0 = 1.967562198638916
        max_targets_1 = 1.967571496963501
        max_targets_2 = 1.967573642730713

        targets[:,0][boundary] *= (1.0 / max_targets_0)
        targets[:,1][boundary] *= (1.0 / max_targets_1)
        targets[:,2][boundary] *= (1.0 / max_targets_2)

        targets = targets.reshape((b, c, h, w))

        return inputs, targets, label