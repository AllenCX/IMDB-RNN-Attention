import numpy as np
import json
import tensorflow as tf
from config import Config
from utils import *
import sys
import os
class imdb_classifier(object):
    def __init__(self, config, session, x_train, y_train, x_test, y_test, train_length, test_lentgh):
        self.config = config
        self.embedding_size = config.embedding_size
        self.batch_size = config.batch_size
        self.encoder_hidden_size = config.encoder_hidden_size
        self.vocab_size = config.vocab_size
        self.lr = config.lr
        self.sess = session
        self.epoch_num = config.epoch_num
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_length = train_length
        self.test_length = test_length
        self.max_length = config.max_length
        self.max_grad_norm = config.max_grad_norm
        self.save_per_epoch = config.save_per_epoch
        self.ckpt_path = config.ckpt_path
        self.test_lentgh = test_length
        self.keep_prob = config.keep_prob
        self.atn_hidden_size = config.atn_hidden_size
        
    def build(self):
        self.global_step = tf.Variable(0, name="global_step")
        self.batch_maxlen = tf.placeholder(dtype=tf.int32, name="batch_maxlen")
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name="output_keep_prob")
        self.encoder_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name="encoder_input")
        self.encoder_input_length = tf.placeholder(shape=(None,), dtype=tf.int32, name="encoder_input_length")
        self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name="label")
        self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -5.0, 5.0), 
                                     dtype=tf.float32,
                                     trainable=False,
                                     name="embedding")
        self.encoder_input_embedded = tf.nn.embedding_lookup(self.embedding, 
                                                             self.encoder_input)
        self.encoder_input_embeded = tf.nn.dropout(self.encoder_input_embedded, keep_prob=self.output_keep_prob)
        
        # bidirectional GRU with dropout
        encoder_fw = tf.contrib.rnn.GRUCell(self.encoder_hidden_size)
        encoder_bw = tf.contrib.rnn.GRUCell(self.encoder_hidden_size)
        self.encoder_fw = tf.contrib.rnn.DropoutWrapper(encoder_fw, output_keep_prob=self.output_keep_prob)
        self.encoder_bw = tf.contrib.rnn.DropoutWrapper(encoder_bw, output_keep_prob=self.output_keep_prob)
        
        # Since time_major == False, output shape should be [batch_size, max_time, ...]
        # run the GRU
        with tf.variable_scope("bi-GRU") as scope:
            ((self.encoder_fw_output, self.encoder_bw_output), 
             (self.encoder_fw_state, self.encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw, 
                                                cell_bw=self.encoder_bw, 
                                                inputs=self.encoder_input_embedded,
                                                sequence_length=self.encoder_input_length,
                                                dtype=tf.float32)
            )

            self.encoder_output = tf.concat((self.encoder_fw_output, self.encoder_bw_output), 2) 
            #[batch_size, max_time, 2 * encoder_hidden_size]
            self.encoder_state = tf.concat((self.encoder_fw_state, self.encoder_bw_state), 1)
        
        # Attention layer
        with tf.variable_scope("attention") as scope:
            self._atn_in = tf.expand_dims(self.encoder_output, axis=2) # [batch_size, max_time, 1, 2 * encoder_hidden_size]
            self.atn_w = tf.Variable(
                tf.truncated_normal(shape=[1, 1, 2 * self.encoder_hidden_size, self.atn_hidden_size], stddev=0.1),
                name="atn_w")
            self.atn_b = tf.Variable(tf.zeros(shape=[self.atn_hidden_size]))
            self.atn_v = tf.Variable(
                tf.truncated_normal(shape=[1, 1, self.atn_hidden_size, 1], stddev=0.1),
                name="atn_b")
            self.atn_activations = tf.nn.tanh(
                tf.nn.conv2d(self._atn_in, self.atn_w, strides=[1,1,1,1], padding='SAME') + self.atn_b)
            self.atn_scores = tf.nn.conv2d(self.atn_activations, self.atn_v, strides=[1,1,1,1], padding='SAME')
            atn_probs = tf.nn.softmax(tf.squeeze(self.atn_scores, [2, 3]))
            _atn_out = tf.matmul(tf.expand_dims(atn_probs, 1), self.encoder_output)
            self.atn_out = tf.squeeze(_atn_out, [1], name="atn_out")
            
        # Output layer
        with tf.variable_scope("output") as scope:
            self.output_w = tf.Variable(
                tf.truncated_normal(shape=(self.encoder_hidden_size*2, 2), stddev=0.1), name="output_w") 
            self.output_b = tf.Variable(tf.zeros(2), name="output_b")

            self.logits = tf.matmul(self.atn_out, self.output_w) + self.output_b
            self.prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels), tf.float32))

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            )
        
        '''self.tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars),
                                          self.max_grad_norm)
        
        self.opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.apply_gradients(zip(grads, self.tvars), global_step=self.global_step)'''
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)
        #self.rmsp_train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)
        
        self.saver = tf.train.Saver()
    def train(self, mode=None, restore=False):
        if mode != "continue":   
            print("Building the model...")
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                self.saver.restore(sess=self.sess, save_path=self.ckpt_path)
        
        print("Starting training...")
        print("%d steps per epoch." % (len(self.x_train) // self.batch_size))
        for epoch in range(self.epoch_num):
            loss_in_epoch = []
            acc_in_epoch = []
            for x_batch, y_batch, input_length in self.minibatches(
                self.x_train, self.y_train, self.train_length, batch_size=self.batch_size, shuffle=True):
                # pad inputs
                x_batch, batch_maxlen = self.padding_sequence(x_batch, self.max_length)
                feed_dict = {
                    self.encoder_input: x_batch,
                    self.encoder_input_length: input_length,
                    self.labels: y_batch,
                    self.output_keep_prob: self.keep_prob,
                    self.batch_maxlen: batch_maxlen
                }
                _, loss, step, acc, pred = sess.run(
                    [self.train_op, self.loss, self.global_step, self.accuracy, self.prediction], feed_dict=feed_dict)
                loss_in_epoch.append(loss)
                acc_in_epoch.append(acc)
                sys.stdout.write("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % (epoch, step, loss, acc))
                sys.stdout.flush()
                
            sys.stdout.write("Epoch %d, Step: %d, Loss: %.4f, Acc: %.4f\r" % 
                             (epoch, step, np.mean(loss_in_epoch), np.mean(acc_in_epoch)))
            sys.stdout.flush()
            print("")
            if (epoch + 1) % self.save_per_epoch == 0:
                self.saver.save(self.sess, "models/bi-lstm-imdb.ckpt")
                self.test(sub_size=5000, restore=False)
                
    
    def test(self, sub_size=None, restore=False):
        if sub_size == None:
            train_sub_size = len(self.y_train)
            test_sub_size = len(self.y_test)
        else:
            train_sub_size = sub_size
            test_sub_size = sub_size 
        
        # build and restore the model
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.ckpt_path)

        acc_list = []
        loss_list = []
        for x_batch, y_batch, input_length in self.minibatches(
            self.x_train[:train_sub_size], self.y_train[:train_sub_size], self.train_length[:train_sub_size], self.batch_size, False):
            x_batch, _ = self.padding_sequence(x_batch, self.max_length)
            feed_dict = {
                self.encoder_input: x_batch,
                self.encoder_input_length: input_length,
                self.labels: y_batch,
                self.output_keep_prob: 1.0
            }
            loss, acc, pred = self.sess.run([self.loss, self.accuracy, self.prediction], feed_dict=feed_dict)
            acc_list.append(acc)
            loss_list.append(loss)
            '''print(pred)
            print(y_batch)
            print(acc, np.mean(pred == y_batch))
            return '''
        print("Test finished on training set! Loss: %.4f, Acc: %.4f" % (np.mean(loss_list), np.mean(acc_list)))
        
        acc_list = []
        loss_list = []
        for x_batch, y_batch, input_length in self.minibatches(
            self.x_test[:test_sub_size], self.y_test[:test_sub_size], self.test_lentgh[:test_sub_size], self.batch_size, False):
            x_batch, _ = self.padding_sequence(x_batch, self.max_length)
            feed_dict = {
                self.encoder_input: x_batch,
                self.encoder_input_length: input_length,
                self.labels: y_batch,
                self.output_keep_prob: 1.0
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            acc_list.append(acc)
            loss_list.append(loss)
        print("Test finished on test set! Loss: %.4f, Acc: %.4f" % (np.mean(loss_list), np.mean(acc_list)))
        
    def predict(self, inputs, restore=False):
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.ckpt_path)

        inputs = self.padding_sequence(inputs)
        inputs_length = np.array([len(seq) for seq in inputs])
        feed_dict = {
             self.encoder_input: inputs,
            self.encoder_input_length: input_length,
            self.output_keep_prob: 1.0
        }
        pred = self.sess.run()
    def minibatches(self, inputs=None, targets=None, input_len=None, batch_size=None, shuffle=True):
        assert len(inputs) == len(targets)
        #assert len(inputs) == len(inputs_length)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt], input_len[excerpt]
    
    def padding_sequence(self, inputs, max_length=None):
        batch_size = len(inputs)
        #assert self.batch_size == batch_size
        if max_length != None:
            if np.max([len(i) for i in inputs]) > max_length:
                maxlen = max_length
            else:
                maxlen = np.max([len(i) for i in inputs])
        else:
            maxlen = np.max([len(i) for i in inputs])
        
        output = np.zeros([batch_size, maxlen], dtype=np.int32) 
        for i, seq in enumerate(inputs):     
            output[i, :len(seq[:maxlen])] = np.array(seq[:maxlen])
        return output, maxlen

if __name__ == "__main__":
    config = Config(batch_size=32, 
                embedding_size=128,
                encoder_hidden_size=64,
                vocab_size=88584,
                lr=0.0005, 
                epoch_num=50,
                save_per_epoch=5,
                max_length=128,
                max_grad_norm=5,
                keep_prob=0.2,
                atn_hidden_size=16,
                ckpt_path="models")
    
    # mkdir models
    if not os.path.exists("models"):
        os.mkdir("models")
        
    x_train, y_train, x_test, y_test, train_length, test_length, wid_dict, id2w = load_imdb_data()

    SAMPLE_SIZE = 25000 # To debug, set to 25000 after debuging
    sess = tf.Session()
    classifier = imdb_classifier(config, 
                       sess, 
                       x_train[:SAMPLE_SIZE], 
                       y_train[:SAMPLE_SIZE], 
                       x_test[:SAMPLE_SIZE], 
                       y_test[:SAMPLE_SIZE], 
                       train_length, 
                       test_length)
    if len(sys.argv) == 1:
        # train a new model
        #tf.reset_default_graph()
        classifier.train()
    else:
        if sys.argv[1] == "test":
            classifier.test()
        elif sys.argv[1] == "continue":
            classifier.train(mode="continue", restore=True)
    sess.close()