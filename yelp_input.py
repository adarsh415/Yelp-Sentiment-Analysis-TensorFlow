import tensorflow as tf
import os


NUM_EXMPL_TRAIN=4600
VOCAB_SIZE=50000


def read_file(filename):

    reader=tf.TextLineReader(skip_header_lines=1)
    key,value=reader.read(filename)
    #tf.Print(value,[key,value])
    record_default=[[1],[1],[1]]
    label,feature,seq_len=tf.decode_csv(value,record_defaults=record_default,field_delim="|")
    #label,feature,seq_len=

    return label,feature,seq_len
    #return value



def _create_batch(text,label,seq_len,min_queue_size,batch_size,shuffle):

    num_processor_threads=4

    if shuffle:
        text,label,seq_len=tf.train.shuffle_batch([text,label,seq_len],
                                                  batch_size=batch_size,
                                                  capacity=min_queue_size+3*batch_size,
                                                  num_threads=num_processor_threads,
                                                  min_after_dequeue=min_queue_size)
    else:
        text,label,seq_len=tf.train.batch([text,label,seq_len],
                                          batch_size=batch_size,
                                          capacity=min_queue_size+3*batch_size,
                                          num_threads=num_processor_threads,
                                          dynamic_pad=True)


    #text = tf.string_split(text).values


    return text,tf.one_hot(tf.range(0,tf.shape(label)[0]),depth=2),seq_len


def generate_batch_train(filename,batch_size):


    filename_queue=tf.train.string_input_producer(filename)
    label,feature,seq_len=read_file(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_size=int(min_fraction_of_examples_in_queue*NUM_EXMPL_TRAIN)

    return _create_batch(feature,label,seq_len,min_queue_size,batch_size,shuffle=False)

def make_dir(name):

    if not os.path.exists(os.path.join(os.getcwd(),name)):
        os.mkdir(os.path.join(os.getcwd(),name))

    else:
        pass



filename_queue=tf.train.string_input_producer(['train1.csv'])
#feature,label,seq_len=generate_batch_train(['train1.csv'],1)
label,feature,seq_len=read_file(filename_queue)

with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    count = 0
    while not coord.should_stop():
        x,y,z=sess.run([label,feature,seq_len])
        print ('feature ',x)
        #print('label ', y)
        #print ('length ',z)
        count += 1
        print('Review Number: ', count)
    coord.request_stop()
coord.join(threads)







