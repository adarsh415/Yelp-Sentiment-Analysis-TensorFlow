import tensorflow as tf
import pandas as pd



NUM_EXMPL_TRAIN=4600

def read_file(filename):

    reader=tf.TextLineReader(skip_header_lines=1)
    key,value=reader.read(filename)
    tf.Print(value,[key,value])
    record_default=[[1],['']]

    label,feature=tf.decode_csv(value,record_defaults=record_default)
    return label ,feature



def _create_batch(text,label,min_queue_size,batch_size,shuffle):

    num_processor_threads=4

    if shuffle:
        text,label=tf.train.shuffle_batch([text,label],
                                          batch_size=batch_size,
                                          capacity=min_queue_size+3*batch_size,
                                          min_after_dequeue=min_queue_size,
                                          num_threads=num_processor_threads
                                          )
    else:
        text,label=tf.train.batch([text,label],
                                  batch_size=batch_size,
                                  capacity=min_queue_size+3*batch_size,
                                  num_threads=num_processor_threads
                                  )
    return text,label


def generate_batch_train(filename,batch_size):


    filename_queue=tf.train.string_input_producer([filename])
    label,feature=read_file(filename_queue)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_size=int(min_fraction_of_examples_in_queue*NUM_EXMPL_TRAIN)

    return _create_batch(feature,label,min_queue_size,batch_size,shuffle=True)




#filename_queue=tf.train.string_input_producer(['train.csv'])
label,feature=generate_batch_train('train.csv',64)

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    count = 0
    while not coord.should_stop():
        y,x=sess.run([label,feature])
        print ('feature ',x)
        print ('label ',y)
        count +=1
        print ('Review Number: ',count)
    coord.request_stop()
coord.join(threads)






