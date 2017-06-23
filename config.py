class Config(object):
    def __init__(self, batch_size, embedding_size, encoder_hidden_size, vocab_size, lr, epoch_num, save_per_epoch, maxlen):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.vocab_size = vocab_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.summary_dir = "summary"
        self.save_per_epoch = save_per_epoch
        self.maxlen=maxlen