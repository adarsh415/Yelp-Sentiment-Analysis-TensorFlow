
import tensorflow as tf



class YelpNetwork(object):

    def __init__(self,
                 learning_rate,
                 vocab_size,
                 hidden_size,
                 embedding_size,
                 n_classes=2
                 ):

        self.input=self._input()
        self.target=self._target(n_classes)
        self.seq_len=self._seq_len()
        self.dropout=self._dropout()
        self.embedding=self._embedding_layer(self.input,embedding_size,vocab_size)
        self.scores=self._score(self.embedding,hidden_size,self.dropout,n_classes,self.seq_len)

        self.predict=self._predict(self.scores)
        self.losses=self._losses(self.scores,self.target)
        self.loss=self._loss(self.losses)
        self.accuracy=self._accuracy(self.predict,self.target)
        self.train=self._train_step(learning_rate,self.loss)
        self.merge=tf.summary.merge_all()

    def _input(self):
        """
            :param max_length: Maximum length of an input tensor
            :return: Input placeholder with shape [batch_size, max_length]
        """

        return tf.placeholder(tf.float32,[None,None],name='input')

    def _seq_len(self):
        """
        :return: Sequence length placeholder with shape [batch_size]. Holds each tensor's real length in a given batch,
                        allowing a dynamic sequence length.
        """

        return tf.placeholder(tf.int32,[None],name='seq_len')

    def _target(self,n_classes):
        """

        :param n_classes:Number of classification classes
        :return:Target placeholder with shape [batch_size, n_classes]
        """
        return tf.placeholder(tf.float32,[None,n_classes],name='target')


    def _dropout(self):

        return tf.placeholder(tf.float32,name='dropout')

    def _embedding_layer(self,x,embedding_size,vocab_size):
        """
        Builds the embedding layer with shape [vocab_size, embedding_size]
        :param x:Input with shape [batch_size, max_length]
        :param embedding_size:Words will be represented using a vector of this size
        :param vocab_size:Vocabulary size (number of possible words that may appear in a sample)
        :return: Embedding lookup tensor with shape [batch_size, max_length, embedding_size]
        """

        with tf.variable_scope('embedding'):
            embed=tf.Variable('embedding',tf.random_uniform([vocab_size,embedding_size],-1.0,1.0,seed=0))
            embedding=tf.nn.embedding_lookup(embed,x)
        return embedding


    def _cell(self,hidden_size,keep_proba):
        """

        :param hidden_size: Number of units in the LSTM cell
        :param keep_proba:Tensor holding the dropout keep probability
        :return:LSTM cell with a dropout wrapper
        """

        with tf.variable_scope('cell'):
            cell=tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
            dropout_cell=tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=keep_proba,output_keep_prob=keep_proba,seed=0)
        return dropout_cell



    def _rnn_cell(self,x,hidden_size,keep_proba,seq_len):
        """
        Builds a LSTM layer
        :param x: Input with shape [batch_size, max_length]
        :param hidden_size: Sequence length tensor with shape [batch_size]
        :param keep_proba: Tensor holding the dropout keep probability
        :return: outputs with shape [batch_size, max_seq_len, hidden_size]
        """

        with tf.variable_scope('rnn_cell'):
            lstm_cell=self._cell(hidden_size,keep_proba)

            output,_=tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32,sequence_length=seq_len)
        return output


    def _score(self, embedding, hidden_size, keep_proba, n_classes,seq_len):

        """
        Builds the LSTM layers and the final fully connected layer
        :param embedding: Embedding lookup tensor with shape [batch_size, max_length, embedding_size]
        :param seq_len: Sequence length tensor with shape [batch_size]
        :param hidden_size: Array holding the number of units in the LSTM cell of each rnn layer
        :param n_classes: Number of classification classes
        :param keep_proba: Tensor holding the dropout keep probability
        :return: Linear activation of each class with shape [batch_size, n_classes]
        """

        #Building LSTM layer

        outputs=embedding
        for h in hidden_size:
            outputs=self._rnn_cell(outputs, h,keep_proba,seq_len)

        # Current shape of outputs: [batch_size, max_seq_len, hidden_size]. Reduce mean on index 1
        outputs=tf.reduce_mean(outputs,reduction_indices=[1])

        # Current shape of outputs: [batch_size, hidden_size]. Build fully connected layer
        with tf.name_scope('final_layer/weight'):
            W=tf.Variable(tf.truncated_normal([hidden_size[-1],n_classes],seed=0))
            self.variable_summary(W,'final_layer/weights')
        with tf.name_scope('final_layer/biases'):
            b=tf.Variable(tf.constant(0.1,shape=[n_classes]))
            self.variable_summary(b,'final_layer/biases')
        with tf.name_scope('final_layer/wx_plus_b'):
            scores=tf.nn.xw_plus_b(outputs,W,b,name='scores')
            tf.summary.histogram('final_layer/wx_plus_b',scores)
        return scores

    def _predict(self,scores):


        with tf.name_scope('final_layer/softmax'):
            softmax=tf.nn.softmax(scores,name='predictions')
            tf.summary.histogram('final_layer/softmax',softmax)
        return softmax

    def _losses(self,scores,target):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Cross entropy losses with shape [batch_size]
        """

        with tf.name_scope('cross_entropy'):
            cross_entropy=tf.nn.softmax_cross_entropy_with_logits(scores,target,name='cross_entropy')
        return cross_entropy


    def _loss(self,losses):
        """
        :param losses: Cross entropy losses with shape [batch_size]
        :return: Cross entropy loss mean
        """

        with tf.name_scope('loss'):
            loss=tf.reduce_mean(losses,name='loss')
            tf.summary.scalar('loss',loss)
        return loss

    def _train_step(self,learning_rate,loss):
        """
        :param learning_rate: Learning rate of RMSProp algorithm
        :param loss: Cross entropy loss mean
        :return: RMSProp train step operation
        """

        return tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    def _accuracy(self,predict,target):
        """
        :param predict: Softmax activations with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Accuracy mean obtained in current batch
        """

        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(predict,1),tf.argmax(target,1))
            accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuaracy')
            tf.summary.scalar('accuracy',accuracy)
        return accuracy

    def initialize_all_variable(self):
        return tf.initialize_all_variables()


    @staticmethod
    def variable_summary(var,name):

        with tf.name_scope('summaries'):
            mean=tf.reduce_mean(var)
            tf.summary.scalar('mean/'+name,mean)

            with tf.name_scope('stddev'):
                stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))

            tf.summary.scalar('stddev/'+name,stddev)
            tf.summary.scalar('max/'+name,tf.reduce_max(var))
            tf.summary.scalar('min/'+name,tf.reduce_min(var))
            tf.summary.histogram(name,var)



