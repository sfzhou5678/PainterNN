import tensorflow as tf
from pnn.model.cnn_utils import *


class BackwardRPNNModel:
  def __init__(self, config, is_training, GO_ID):
    self.input_ids = tf.placeholder(tf.int32, [None, config.max_num_steps], name='input_ids')
    self.labels = tf.placeholder(tf.int32, [None, config.n_classes], name='labels')

    self.config = config
    self.is_training = is_training

    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable("embedding", [config.vocab_size, config.word_embedding_size], dtype=tf.float32,
                                       trainable=False,
                                       )
      self.embedd_encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.input_ids)
      # if is_training:
      #   rnn_embed_inputs = tf.nn.dropout(self.embedd_encoder_inputs, config.keep_prob)
      # else:
      rnn_embed_inputs = self.embedd_encoder_inputs

    self.embedded_GO_ID = tf.nn.embedding_lookup(self.embedding,
                                                 tf.constant(GO_ID, dtype=tf.int32,
                                                             shape=[config.batch_size],
                                                             name='GO_ID'))
    self.pixel_vec = self.get_pixel_vec(rnn_embed_inputs, config.hidden_size)

    self.logits = self.cnn_clf(self.pixel_vec, 'cnn_encoder', not is_training)

    self.loss, self.train_op = self.build_loss_trin_op(self.logits, self.labels)

    correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, axis=1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def get_pixel_vec(self, rnn_embed_inputs, hidden_units):
    encoder_outputs, encoded_final_state = self.pnn_encode(rnn_embed_inputs, self.is_training)
    pixel_vec = self.pnn_decode(encoder_outputs, encoded_final_state, hidden_units, self.is_training)

    return pixel_vec

  def pnn_encode(self, embedding_inputs, is_training, rnn_layers=2, keep_prob=0.5):
    with tf.variable_scope('encoder', reuse=(not self.is_training)) as encoder_scope:
      def build_cell(hidden_size):
        def get_single_cell(hidden_size, keep_prob):
          cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
          if is_training and keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
          return cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
          [get_single_cell(hidden_size, keep_prob) for _ in range(rnn_layers)])

        return cell

      if not self.config.bi_lstm:
        encoder_cell = build_cell(self.config.hidden_size)
        # Run Dynamic RNN#   encoder_outpus: [max_time, batch_size, num_units]#   encoder_state: [batch_size, num_units]
        # TODO 这里的sequence_length表示input_sequence_length，即原始输入中的非PAD的长度(所以会跳过PAD不训练)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
          encoder_cell, embedding_inputs,
          sequence_length=tf.constant(self.config.max_num_steps, shape=[self.config.batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)
        return encoder_outputs, encoder_final_state
      else:
        encoder_cell = build_cell(self.config.hidden_size / 2)
        bw_encoder_cell = build_cell(self.config.hidden_size / 2)
        encoder_outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
          encoder_cell, bw_encoder_cell,
          embedding_inputs,
          sequence_length=tf.constant(self.config.max_num_steps, shape=[self.config.batch_size], dtype=tf.int32),
          dtype=tf.float32, scope=encoder_scope)

        state = []
        for i in range(rnn_layers):
          fs = fw_state[i]
          bs = bw_state[i]
          encoder_final_state_c = tf.concat((fs.c, bs.c), 1)
          encoder_final_state_h = tf.concat((fs.h, bs.h), 1)
          encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h)
          state.append(encoder_final_state)
        encoder_final_state = tuple(state)

        encoder_outputs = tf.maximum(encoder_outputs[0], encoder_outputs[1])
        return encoder_outputs, encoder_final_state

  def pnn_decode(self, encoder_outputs, encoder_final_state, hidden_units, is_training, rnn_layers=2,
                 keep_prob=0.5):
    with tf.variable_scope("decoder", ):

      def build_decoder_cell(memory, use_attention=False):
        def get_single_cell(hidden_size, keep_prob):
          cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
          if is_training and keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
          return cell

        # 如果只有decoder用了MultiRNNCell而encoder用的是BasicCell那么就会报错(不一致就会报错)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
          [get_single_cell(hidden_units, keep_prob) for _ in range(rnn_layers)])
        decoder_init_state = encoder_final_state

        if use_attention:
          ## Create an attention mechanism
          memory = memory  # TODO 在反向attention中，memory应该是pixelVecs，并且这个memory应该是动态的
          attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            self.config.hidden_size, memory,
            # memory_sequence_length=seq_lengths,
            memory_sequence_length=tf.constant(self.config.max_num_steps, shape=[self.config.batch_size],
                                               dtype=tf.int32))

          # alignment_history = tf.cond(is_training, lambda: False, lambda: True)
          decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=self.config.hidden_size,
            # alignment_history=False,  # test时为true
            name="attention")

          attention_states = decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(
            cell_state=encoder_final_state)
          decoder_init_state = attention_states

        return decoder_cell, decoder_init_state

      last_pixel_vec = tf.zeros(
        [self.config.batch_size, self.config.width, self.config.height, self.config.word_embedding_size],
        dtype=tf.float32)
      pixel_vec_list = []
      pixel_vec_list.append(last_pixel_vec)

      decoder_cell, decoder_init_state = build_decoder_cell(pixel_vec_list, use_attention=False)
      state = decoder_init_state

      # fixme 不知道pixelList的更新能不能传递到attention层，如果不能的话需要手动初始化
      with tf.variable_scope("RNN"):
        # 为句子中的每一个单词建模,共执行maxNumSteps次
        for time_step in range(self.config.max_num_steps):
          if time_step > 0:
            tf.get_variable_scope().reuse_variables()
          last_pixel_latent = self.get_latent(last_pixel_vec, self.config.word_embedding_size, self.config.hidden_size,
                                              self.is_training, name='abc')

          (increment_weights, state) = decoder_cell(last_pixel_latent, state)
          increment_weights = tf.reshape(increment_weights,
                                         [self.config.batch_size, self.config.width, self.config.height])

          cur_word_embedding = self.embedd_encoder_inputs[:, time_step, :]
          increment_vec = increment_weights[:, :, :, None] * cur_word_embedding[:, None, None, :]

          last_pixel_vec += increment_vec
          pixel_vec_list.append(last_pixel_vec)
      final_pixel_vec = last_pixel_vec

      return final_pixel_vec

  def cnn_clf(self, pixel_vec, name, reuse):
    norm = True

    # TODO 构建网络的过程弄成for
    DEPTH1 = 32
    DEPTH2 = DEPTH1 * 2
    DEPTH3 = DEPTH2 * 2
    DEPTH4 = DEPTH3 * 2

    with tf.variable_scope(name, reuse=reuse):
      # 第一层
      network = conv_2d(pixel_vec, [3, 3, self.config.word_embedding_size, DEPTH1], [DEPTH1], [1, 1, 1, 1],
                        'layer1-conv1',
                        norm=norm,
                        is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH1, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer1-pool1')

      # 第二层
      network = conv_2d(network, [3, 3, DEPTH1, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv1',
                        norm=norm, is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH2, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer2-pool1')

      # 第三层
      network = conv_2d(network, [3, 3, DEPTH2, DEPTH3], [DEPTH3], [1, 1, 1, 1], 'layer3-conv1', norm=norm,
                        is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH3, DEPTH3], [DEPTH3], [1, 1, 1, 1], 'layer3-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer3-pool1')

      # 第四层
      network = conv_2d(network, [3, 3, DEPTH3, DEPTH4], [DEPTH4], [1, 1, 1, 1], 'layer4-conv1', norm=norm,
                        is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH4, DEPTH4], [DEPTH4], [1, 1, 1, 1], 'layer4-conv2',
                        norm=norm, is_training=self.is_training)
      network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer4-pool1')

      # 最后将CNN产生的值通过全局平均池化，再通过全连接层产生latent vector
      net = slim.avg_pool2d(network, network.get_shape()[1:3], padding='VALID', scope='AvgPool')
      # 这里不能加is_training=false，如果加了就会导致val时所有cos均为1 (原因未知，但是官方IncepResnetV2中也是恒为true的)
      net = slim.dropout(net, 0.5, is_training=self.is_training, scope='Dropout')
      net = slim.flatten(net)

      # 最后加上一个全连接层做分类
      logits = slim.fully_connected(net, self.config.n_classes, activation_fn=None, scope='latent_vec')

      return logits

  def build_loss_trin_op(self, logits, labels):
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    loss = cross_entropy_mean  # 没有加正则化

    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
      0.01,
      global_step,
      300,
      0.98,
      staircase=True
    )
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

    return loss, train_op

  def assign_embedding(self, sess, word_vector):
    sess.run(tf.assign(self.embedding, word_vector))

  def get_latent(self, input, input_depth, output_size, is_training, name):
    norm = True

    # TODO 构建网络的过程弄成for
    DEPTH1 = 32
    DEPTH2 = DEPTH1 * 2
    DEPTH3 = DEPTH2 * 2
    DEPTH4 = DEPTH3 * 2

    with tf.variable_scope(name):
      # 第一层
      network = conv_2d(input, [3, 3, input_depth, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv1',
                        norm=norm,
                        is_training=self.is_training)
      network = conv_2d(network, [3, 3, DEPTH1, DEPTH1], [DEPTH1], [1, 1, 1, 1], 'layer1-conv2',
                        norm=norm, is_training=self.is_training)
      # 根据阿里2017ICCV论文，这里不加池化层
      # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer1-pool1')

      # 第二层
      # network = conv_2d(network, [3, 3, DEPTH1, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv1',
      #                   norm=norm, is_training=self.is_training)
      # network = conv_2d(network, [3, 3, DEPTH2, DEPTH2], [DEPTH2], [1, 1, 1, 1], 'layer2-conv2',
      #                   norm=norm, is_training=self.is_training)
      # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer2-pool1')

      # # 第三层
      # network = conv_2d(network, [3, 3, DEPTH2, DEPTH3], [DEPTH3], [1, 1, 1, 1], 'layer3-conv1', norm=norm,
      #                   is_training=self.is_training)
      # network = conv_2d(network, [3, 3, DEPTH3, DEPTH3], [DEPTH3], [1, 1, 1, 1], 'layer3-conv2',
      #                   norm=norm, is_training=self.is_training)
      # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer3-pool1')
      #
      # # 第四层
      # network = conv_2d(network, [3, 3, DEPTH3, DEPTH4], [DEPTH4], [1, 1, 1, 1], 'layer4-conv1', norm=norm,
      #                   is_training=self.is_training)
      # network = conv_2d(network, [3, 3, DEPTH4, DEPTH4], [DEPTH4], [1, 1, 1, 1], 'layer4-conv2',
      #                   norm=norm, is_training=self.is_training)
      # network = max_pool_2d(network, [1, 2, 2, 1], [1, 2, 2, 1], 'layer4-pool1')

      # 最后将CNN产生的值通过全局平均池化，再通过全连接层产生latent vector
      net = slim.avg_pool2d(network, network.get_shape()[1:3], padding='VALID', scope='AvgPool')
      # 这里不能加is_training=false，如果加了就会导致val时所有cos均为1 (原因未知，但是官方IncepResnetV2中也是恒为true的)
      net = slim.dropout(net, 0.5, is_training=is_training, scope='Dropout')
      net = slim.flatten(net)

      # 最后加上一个全连接层做分类
      latent = slim.fully_connected(net, output_size, activation_fn=None, scope='latent_vec')

    return latent
