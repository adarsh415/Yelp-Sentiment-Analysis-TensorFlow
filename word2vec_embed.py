



import tensorflow as tf
import numpy as np
from process_data import process_data
import os

vocabulary_size=50000  # size of vocabulary
embedding_size=300   # size of embedding
batch_size=64  # batch size
skip_window=1  # How many words to consider left and right
num_skip=2  # How many time to reuse imput to generate label
negative_sample=64 #how many negative example to sample
LEARNING_RATE=1.0
NUM_TRAIN_STEPS=100000
SKIP_STEP = 1000 # how many steps to skip before reporting the loss

def word2vec(batch_gen):

    graph=tf.Graph()

    with graph.as_default():

        # input data.
        with tf.name_scope('input'):
            center_words=tf.placeholder(tf.int32,shape=[batch_size],name='center_word')
            target_words=tf.placeholder(tf.int32,shape=[batch_size,1],name='target_words')

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):

            # lookup for embedding for input

            with tf.name_scope('embedding'):
                embedding=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
                embed=tf.nn.embedding_lookup(embedding,center_words)

            #Construct variable to NCE loss

            with tf.name_scope('weights'):

                nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size], stddev=1.0 / np.math.sqrt(embedding_size)))

            with tf.name_scope('biases'):
                biases=tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.

        with tf.name_scope('loss'):
            loss=tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=biases,
                    inputs=embed,
                    labels=target_words,
                    num_sampled=negative_sample,
                    num_classes=vocabulary_size,name='loss' )
            )

        #tf.summary.scalar('loss',loss)

        optimizer=tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
        make_dir('checkpoints')


    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())

        total_loss=0.0 #we use this to calculate avarage loss
        writer=tf.summary.FileWriter('./graphs/',sess.graph)
        # Create model saving operation
        saver=tf.train.Saver({"embedding":embedding})

        for index in range(NUM_TRAIN_STEPS):
            centers,targets=next(batch_gen)

            loss_batch,_=sess.run([loss,optimizer],feed_dict={center_words:centers,target_words:targets})

            total_loss +=loss_batch

            if (index+1)% SKIP_STEP == 0:

                embedding_name= './'+'checkpoints'+'/'+'yulp_review_step_'+str(index)+'.ckpt'
                model_checkpoint_path=os.path.join(os.getcwd(),embedding_name)
                save_path=saver.save(sess,model_checkpoint_path)
                print ('embedding saved in {}'.format(save_path))

                print ('Avg loss at step {}: {:5.1f}'.format(index,total_loss/SKIP_STEP))
                total_loss=0.0
        writer.close()


def make_dir(name):

    if not os.path.exists(os.path.join(os.getcwd(),name)):
        os.mkdir(os.path.join(os.getcwd(),name))

    else:
        pass


def main():

    batch_gen=process_data(vocabulary_size,batch_size,skip_window)
    word2vec(batch_gen)

if __name__=='__main__' :
    main()




