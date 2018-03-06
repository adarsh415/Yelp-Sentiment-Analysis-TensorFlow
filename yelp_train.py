
import tensorflow as tf
import yelp_input as yp
import yelp_netowrk as ynn



lstn_n=ynn.YelpNetwork(learning_rate=0.01,
                       vocab_size=50000,
                       hidden_size=[32,32],
                       embedding_size=300)
yp.make_dir('summary_writer')
train_writer=tf.summary.FileWriter('.\summary_writer')
saver=tf.train.Saver()

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    sess.run(lstn_n.initialize_all_variable())
    sess.run(tf.global_variables_initializer())
    xtrain, ytrain,seq_len = yp.generate_batch_train('train.csv', 64)
    count =1
    while not coord.should_stop():


        train_loss,_,summary=sess.run([lstn_n.loss,lstn_n.train,lstn_n.merge],feed_dict={lstn_n.input:xtrain,lstn_n.target:ytrain,lstn_n.dropout:0.4,lstn_n.seq_len:seq_len})

        train_writer.add_summary(summary, count)  # Write train summary for step count(TensorBoard visualization)
        print('{0} train loss: {1:.4f}'.format( count, train_loss))
        count+=1


    saver.save('.\checkpoints\model.ckpt')
    coord.request_stop()
coord.join(threads)