from pnn.model.BackwardRPNNModel import BackwardRPNNModel

import numpy as np
import tensorflow as tf
from pnn.data_for_model import tfdata_handler
from pnn.config import SmallConfig
from collections import Counter

slim = tf.contrib.slim


def mini_scale_word_vector(id_sequences, wordVectors, max_word_count):
  assert len(wordVectors) >= max_word_count

  id_counter = Counter(id_sequences.flatten())
  id_counter = sorted(id_counter.items(), key=lambda d: d[1], reverse=True)

  count = 0
  useful_word_dic = {}
  new_word_vectors = []
  for key, value in id_counter:
    useful_word_dic[key] = count
    new_word_vectors.append(wordVectors[key])
    count += 1
    if count >= max_word_count:
      break

  for i in range(len(wordVectors)):
    if count >= max_word_count:
      break
    if i not in useful_word_dic:
      useful_word_dic[i] = count
      new_word_vectors.append(wordVectors[i])
      count += 1

  for sentence, j in zip(id_sequences, range(len(id_sequences))):
    for id, i in zip(sentence, range(len(sentence))):
      if id not in useful_word_dic:
        id_sequences[j][i] = useful_word_dic[399999]
      else:
        id_sequences[j][i] = useful_word_dic[id]
  return id_sequences, np.array(new_word_vectors)


PAD = 0
GO_ID = 1
END_ID = 2


def num_train():
  # wordsList = np.load('data/wordsList.npy').tolist()
  # wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
  wordVectors = np.load('data/wordVectors.npy')
  # print(wordVectors)
  print('Reading word vector completed.')

  config = SmallConfig()

  mini_scale = True
  id_sequences = np.load('data/idsMatrix.npy')
  # id_sequences = id_sequences[12050:12950]
  id_sequences = np.array([sentence[:config.max_num_steps] for sentence in id_sequences])

  if mini_scale:
    id_sequences, wordVectors = mini_scale_word_vector(id_sequences, wordVectors, config.vocab_size)

  labels = [0 if i <= (len(id_sequences) - 1) // 2 else 1 for i in range(len(id_sequences))]
  tfdata_handler.prepare_tfrecords(id_sequences, labels)

  # # 这种tfRecords的方式也可以用PTB中的data_producer代替
  train_ids, train_label = tfdata_handler.get_records('data/train.tfrecords', config.max_num_steps, config.n_classes)
  train_ids_batch, train_label_batch = tf.train.shuffle_batch([train_ids, train_label], batch_size=config.batch_size,
                                                              capacity=config.batch_size * 3 + 1000,
                                                              min_after_dequeue=config.batch_size * 3 + 1000 - 1)

  val_ids, val_label = tfdata_handler.get_records('data/valid.tfrecords', config.max_num_steps, config.n_classes)
  val_ids_batch, val_label_batch = tf.train.shuffle_batch([val_ids, val_label], batch_size=config.batch_size,
                                                          capacity=config.batch_size * 3 + 1000,
                                                          min_after_dequeue=config.batch_size * 3 + 1000 - 1)
  print('Building data batch completed.')

  initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
  with tf.name_scope('Train'):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      train_model = BackwardRPNNModel(config=config, is_training=True, GO_ID=GO_ID)

  with tf.name_scope('Valid'):
    with tf.variable_scope("Model", reuse=True):
      val_model = BackwardRPNNModel(config=config, is_training=False, GO_ID=GO_ID)

  print('Building models completed.')

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_model.assign_embedding(sess, wordVectors)
    print('Assigning WordVectors complected.')

    print('===========training===========')
    need_val = True
    for i in range(20000):
      train_ids, train_labels = sess.run([train_ids_batch, train_label_batch])

      _, loss, acc = sess.run([train_model.train_op, train_model.loss,
                               train_model.accuracy],
                              {train_model.input_ids: train_ids, train_model.labels: train_labels})

      if i % 100 == 0:
        if need_val:
          if i > 600:
            total_val_loss = 0
            total_val_acc = 0
            n = 300 // config.batch_size
            for _ in range(n):
              val_ids, val_labels = sess.run([val_ids_batch, val_label_batch])

              val_loss, val_acc = sess.run([val_model.loss, val_model.accuracy],
                                           {val_model.input_ids: val_ids, val_model.labels: val_labels})
              total_val_loss += val_loss
              total_val_acc += val_acc
            print('[%d]' % i, loss, acc, total_val_loss / n, total_val_acc / n)
          else:
            val_ids, val_labels = sess.run([val_ids_batch, val_label_batch])

            val_loss, val_acc = sess.run([val_model.loss, val_model.accuracy],
                                         {val_model.input_ids: val_ids, val_model.labels: val_labels})
            print('[%d]' % i, loss, acc, val_loss, val_acc)
        else:
          print('[%d]' % i, loss, acc)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  num_train()
