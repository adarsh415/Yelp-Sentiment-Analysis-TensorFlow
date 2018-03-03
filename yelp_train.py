
import tensorflow as tf
import process_data as pd
import yelp_input as yp
import yelp_netowrk as ynn







xtrain,ytrain=yp.generate_batch_train('train.csv',64)
drop_out=tf.placeholder(tf.float32,name='drop_out')

lstn_n=ynn.YelpNetwork(max_len=xtrain,
                       learning_rate=0.01,
                       vocab_size=50000,
                       hidden_size=[32,32],
                       embedding_size=300,
                       keep_proba=drop_out,
                       target=ytrain)
yp.make_dir('summary_writer')
train_writer=tf.summary.FileWriter('.\summary_writer')
saver=tf.train.Saver()

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    sess.run(lstn_n.initialize_all_variable())
    count =1
    while not coord.should_stop():

        train_loss,_,summary=sess.run([lstn_n.loss,lstn_n.train,lstn_n.merge],feed_dict={drop_out:0.4})

        train_writer.add_summary(summary, count)  # Write train summary for step i (TensorBoard visualization)
        print('{0} train loss: {1:.4f}'.format( count, train_loss))
        count+=1


    saver.save('.\checkpoints\model.ckpt')
    coord.request_stop()
coord.join(threads)