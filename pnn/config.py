class SmallConfig:
  # 整体相关
  batch_size = 8
  n_classes = 2

  max_num_steps = 200
  vocab_size = 20000

  keep_prob = 0.5
  base_learning_rate = 0.01
  lr_decay = 0.98
  init_scale = 0.1

  # CNN相关(现在直接写在代码里面了)
  width=22
  height=22

  # RNN相关
  word_embedding_size = 50
  # hidden_size = 128
  hidden_size = width*height

  rnn_layers = 2
  bi_lstm = True

