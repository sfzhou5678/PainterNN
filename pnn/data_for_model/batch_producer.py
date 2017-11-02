# import tensorflow as tf
# import numpy as np
# import random
#
#
# def batch_producer(raw_data, batch_size, num_steps, PAD, name=None):
#   with tf.name_scope(name, "BatchProducer", [raw_data, batch_size, num_steps]):
#     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
#     raw_data = tf.reshape(raw_data, [-1])
#
#     data_len = tf.size(raw_data)
#     batch_len = data_len // batch_size
#     data = tf.reshape(raw_data[0: batch_size * ((data_len // batch_size // num_steps) * num_steps)],
#                       [batch_size, ((data_len // batch_size // num_steps) * num_steps)])
#
#     epoch_size = (batch_len - 1) // num_steps
#     assertion = tf.assert_positive(
#       epoch_size,
#       message="epoch_size == 0, decrease batch_size or num_steps")
#     with tf.control_dependencies([assertion]):
#       epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#     x = tf.strided_slice(data, [0, i * num_steps],
#                          [batch_size, (i + 1) * num_steps])
#     x.set_shape([batch_size, num_steps])
#
#     t = tf.not_equal(x, tf.constant(PAD))
#     seq_length = tf.reduce_sum(tf.cast(t, dtype=tf.int32), axis=-1)
#
#     return x, seq_length
#
#
# def get_text_data_batch(config, PAD):
#   def rescale_data(raw_data, max_num_steps, padding_mark):
#     padded_data = []
#     for cur_data in raw_data:
#       if len(cur_data) < max_num_steps:
#         padding_data = [padding_mark] * (max_num_steps - len(cur_data))
#         cur_data = np.append(cur_data, padding_data)
#       else:
#         cur_data = cur_data[:max_num_steps]
#       padded_data.append(cur_data)
#
#     return np.array(padded_data)
#
#   def remap_wordvector(id_sequences, wordVectors, max_word_count, old_unk_id=399999, ):
#     assert len(wordVectors) >= max_word_count
#     from collections import Counter
#     id_counter = Counter(id_sequences.flatten())
#     id_counter = sorted(id_counter.items(), key=lambda d: d[1], reverse=True)
#
#     useful_word_dic = {}
#     new_word_vectors = []
#
#     # FIXME: 这里写死了
#     pad_id = 0
#     go_id = 1
#     unk_id = 2
#     #
#     useful_word_dic[pad_id] = pad_id
#     new_word_vectors.append(wordVectors[pad_id])
#
#     # go_id不存在于原数据中，所以不需要映射
#     new_word_vectors.append(np.zeros_like(wordVectors[pad_id]))  # go_id
#
#     useful_word_dic[old_unk_id] = unk_id
#     new_word_vectors.append(np.zeros_like(wordVectors[pad_id]))  # unk_id
#
#     count = 3  # 0,1,2分别被pad、go、unk占用了
#     for key, value in id_counter:
#       if key in useful_word_dic:
#         continue
#       useful_word_dic[key] = count
#       new_word_vectors.append(wordVectors[key])
#       count += 1
#       if count >= max_word_count:
#         break
#
#     # 处理newWrodVec不满maxWordCount的情况
#     for i in range(len(wordVectors)):
#       if count >= max_word_count:
#         break
#       if i not in useful_word_dic:
#         useful_word_dic[i] = count
#         new_word_vectors.append(wordVectors[i])
#         count += 1
#
#     for sentence, j in zip(id_sequences, range(len(id_sequences))):
#       for id, i in zip(sentence, range(len(sentence))):
#         if id not in useful_word_dic:
#           id_sequences[j][i] = unk_id
#         else:
#           id_sequences[j][i] = useful_word_dic[id]
#
#     # 统计一下词频
#     id_counter = Counter(id_sequences.flatten())
#     total_words = sum(id_counter.values())
#     padding_words = id_counter[0]
#     unk_words = id_counter[unk_id]
#     print('total=%d padding=%d,%.1f%% unk=%d,%.1f%%' % (total_words, padding_words, padding_words / total_words * 100.0,
#                                                         unk_words, unk_words / total_words * 100.0))
#
#     return id_sequences, np.array(new_word_vectors)
#
#   wordVectors = np.load('data/wordVectors.npy')
#   id_sequences = np.load('data/idsMatrix.npy')
#   # id_sequences = id_sequences[10050:14950]
#
#   # 先根据max_steps缩放数据(注意这里的padding_mark还是old的399999)
#   id_sequences = rescale_data(id_sequences, config.max_num_steps, padding_mark=0)
#   id_sequences, wordVectors = remap_wordvector(id_sequences, wordVectors, config.vocab_size, old_unk_id=399999)
#
#   training_data = id_sequences[:int(len(id_sequences) * 0.8)]
#   test_data = id_sequences[int(len(id_sequences) * 0.8):]
#
#   training_encinput_batch, training_encinput_seqlen_batch = batch_producer(training_data, config.batch_size,
#                                                                            config.max_num_steps, PAD=PAD)
#   test_encinput_batch, test_encinput_seqlen_batch = batch_producer(test_data, config.batch_size, config.max_num_steps,
#                                                                    PAD=PAD)
#
#   return training_encinput_batch, training_encinput_seqlen_batch, \
#          test_encinput_batch, test_encinput_seqlen_batch
#
#
# def get_num_data_batch(config, GO_ID, END_ID, PAD):
#   def build_raw_data(max_num_steps, vocab_size, how_many, END_ID, GO_ID):
#     raw_encoder_data = []
#     raw_decoder_data = []
#     for i in range(how_many):
#       length = random.randint(3, max_num_steps - 1)
#       encoder_data = np.random.randint(3, vocab_size - 1, [length])
#       encoder_data = np.append(encoder_data, END_ID)
#       raw_encoder_data.append(encoder_data)
#
#       decoder_data = np.append(GO_ID, encoder_data)
#       raw_decoder_data.append(decoder_data)
#
#     return np.array(raw_encoder_data), np.array(raw_decoder_data)
#
#   def build_padded_data(raw_data, max_num_steps, padding_mark):
#     padded_data = []
#     for cur_data in raw_data:
#       padding_data = [padding_mark] * (max_num_steps - len(cur_data))
#       cur_data = np.append(cur_data, padding_data)
#       padded_data.append(cur_data)
#
#     return np.array(padded_data)
#
#   raw_encoder_input_data, raw_decoder_input_data = build_raw_data(config.max_num_steps, config.vocab_size, config.howmany,
#                                                                   END_ID=END_ID, GO_ID=GO_ID)
#
#   padded_encoder_inputs = build_padded_data(raw_encoder_input_data, config.max_num_steps, padding_mark=0)
#   training_encoder_inputs = padded_encoder_inputs[:int(len(padded_encoder_inputs) * 0.8)]
#   test_encoder_inputs = padded_encoder_inputs[int(len(padded_encoder_inputs) * 0.8):]
#   training_encoder_inputs_batch, training_source_seq_length_batch = batch_producer(training_encoder_inputs,
#                                                                                    config.batch_size,
#                                                                                    config.max_num_steps, PAD)
#   test_encoder_inputs_batch, test_source_seq_length_batch = batch_producer(test_encoder_inputs, config.batch_size,
#                                                                            config.max_num_steps,
#                                                                            PAD)
#   ## decoder相关的data，暂时用不到
#   # padded_decoder_inputs = build_padded_data(raw_decoder_input_data, config.max_num_steps + 1, padding_mark=0)
#   # training_decoder_targets = padded_decoder_inputs[:int(len(padded_decoder_inputs) * 0.8)]
#   # test_decoder_targets = padded_decoder_inputs[int(len(padded_decoder_inputs) * 0.8):]
#   # training_decoder_targets_batch, training_targets_seq_length_batch = batch_producer(training_decoder_targets,
#   #                                                                                    config.batch_size,
#   #                                                                                    config.max_num_steps + 1, PAD)
#   # test_decoder_targets_batch, _ = batch_producer(test_decoder_targets,
#   #                                                config.batch_size,
#   #                                                config.max_num_steps + 1, PAD)
#
#   return training_encoder_inputs_batch, training_source_seq_length_batch, \
#          test_encoder_inputs_batch, test_source_seq_length_batch
