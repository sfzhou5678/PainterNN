import random
import tensorflow as tf


def get_example(id_sequence, label):
  example = tf.train.Example(features=tf.train.Features(feature={
    'id_sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=id_sequence)),
    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
  }))

  return example


def prepare_tfrecords(id_sequences, labels):
  examples = [get_example(id_sequences[i], labels[i]) for i in range(len(id_sequences))]
  random.shuffle(examples)

  examples_dic = {}
  examples_dic['train'] = examples[:int(len(examples) * 0.8)]
  examples_dic['valid'] = examples[int(len(examples) * 0.8):int(len(examples) * 0.9)]
  examples_dic['test'] = examples[int(len(examples) * 0.9):]

  for type in ['train', 'valid', 'test']:
    writer = tf.python_io.TFRecordWriter("data/%s.tfrecords" % type)
    for example in examples_dic[type]:
      writer.write(example.SerializeToString())
    writer.close()


def get_records(record_path,max_num_steps,n_classes):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([record_path])

  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      "id_sequence": tf.FixedLenFeature([max_num_steps], tf.int64),
      "label": tf.FixedLenFeature([], tf.int64)
    })
  id_sequence = tf.cast(features['id_sequence'], tf.int32)
  label = tf.cast(features['label'], tf.int32)
  label = tf.one_hot(label, n_classes)

  return id_sequence, label


def data_producer(id_sequences, labels, batch_size, num_steps, name=None):
  with tf.name_scope(name, "DataProducer"):
    id_sequences = tf.convert_to_tensor(id_sequences, name="raw_data", dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, name="raw_label", dtype=tf.int32)

    data_len = tf.shape(id_sequences)[0]  # 25000
    batch_len = data_len // batch_size  # 25000//batchsize
    # 数据取整，最后多余的一点就扔掉了
    # data = tf.reshape(id_sequences[0: batch_size * batch_len, :],
    #                   [batch_size, batch_len, -1])
    data = id_sequences[0:batch_size * batch_len, :]
    reshaped_labels = labels[0:batch_size * batch_len]
    # reshaped_labels = tf.reshape(labels[0: batch_size * batch_len],
    #                              [batch_size, batch_len])

    epoch_size = batch_len
    # assertion = tf.assert_positive(
    #   epoch_size,
    #   message="epoch_size == 0, decrease batch_size or num_steps")
    # # TODO tf.identity??
    # with tf.control_dependencies([assertion]):
    #   epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [i, 0],
                         [batch_size, (i + 1)])
    y = tf.strided_slice(reshaped_labels, [i],
                         [batch_size, (i + 1)])
    # x.set_shape([batch_size, num_steps])
    # y = tf.strided_slice(data, [0, i * num_steps + 1],
    #                      [batch_size, (i + 1) * num_steps + 1])
    # y.set_shape([batch_size, num_steps])
    return x, y
