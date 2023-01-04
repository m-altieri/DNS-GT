import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Cannot enable memory growth on first physical device.')
    sys.exit(1)

import numpy as np
import argparse
import random
from models import NSRModel
from models import DELM
import time
import os
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import logging


# def load_nodes():
#     hosts = np.load('arrays/hosts.npy', allow_pickle=True)
#     domains = np.load('arrays/domains.npy', allow_pickle=True)

#     return hosts, domains


def get_logger(verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger
    
   
def build_model(model, H=None, D=None, seqlen=None):
    if model == 'nsr':
        model = NSRModel(H, D)
    elif model == 'delm':
        model = DELM()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)],
                  run_eagerly=False)

    return model
   

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--es', action='store_const', const=True, help='Early Stopping')
    argparser.add_argument('--load', action='store_const', const=True, help='Keep training the model from the last checkpoint')
    argparser.add_argument('-v', '--verbose', action='store_const', const=True, help='Log debug information')
    argparser.add_argument('--epochs', action='store', default=10, type=int, help='Number of training epochs')
    argparser.add_argument('--bs', action='store', default=128, type=int, help='Batch size')
    argparser.add_argument('--debug', action='store_const', const=True, help='Used for debugging purposes')
    argparser.add_argument('--tf-data-api', action='store_const', const=True, help='Use the TF Data API as data pipeline')
    argparser.add_argument('--seqlen', action='store', default=10, type=int, help='Maximum sequence length')
    argparser.add_argument('--stride', action='store', default=1, type=int, help='Stride between sequences (how many queries to shift by)')
    argparser.add_argument('--include-start', action='store_const', const=True, help='Whether to include <START> as the first token of each sequence (total length is unaffected)')
    
    args = argparser.parse_args()
    return args
   
   
def main_NSRModel():
    args = parse_args()
   
    hosts, domains = load_nodes()
    queries = load_queries()
   
    H, D = len(hosts), len(domains)
   
    model = build_model(H, D)
   
    inputs = np.random.rand(H, 32)
    inputs = np.array([inputs for i in range(1000)])
    inputs = np.apply_along_axis(lambda x: x + random.uniform(-.25, .25) * np.std(x), 0, inputs)
    print(np.shape(inputs))

    callbacks = []
    if args.es:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, patience=10))
        print('Using early stopping.')
      
    model.fit(x=inputs, y=inputs, batch_size=16, epochs=10, callbacks=callbacks)
   
    print(inputs[0])
    out = model(inputs[0])
    print(out)

   
def seq_generator_from_folder(input_folder, seqlen, stride=1, include_start=False):
    """Folder containing .npy files, each representing a matrix of shape (n_queries, 2).
    """
    for f in os.listdir(input_folder):
        seqs = create_sequences(os.path.join(input_folder, f), seqlen, stride, include_start)
        for seq in seqs:
            yield seq
            
            
def create_sequences(input_file, seqlen, stride=1, include_start=False):
    # input [queries, 2]
    # output [queries - stride, seqlen, 2]
    
    actual_seqlen = seqlen if not include_start else seqlen - 1

    queries = np.load(input_file)
    
    seqs = np.empty(shape=((len(queries) - actual_seqlen) // stride + 1, seqlen, 2),
                    dtype=object)

    for i in range(len(seqs)):
        if include_start:
            seqs[i][0] = ['<START>', '<START>']
        seqs[i][1:] = queries[i*stride : i*stride+actual_seqlen]

    return seqs
    
'''
def backup_create_sequences(input_file, seqlen, stride=1, include_start=False):
    # input [queries, 2]
    # output [queries - stride, seqlen, 2]
    queries = np.load(input_file)
    
    seqs = np.empty(shape=((len(queries) - seqlen + 1) // stride, seqlen, 2),
                    dtype=object)

    for i in range(len(seqs)):
        seqs[i] = queries[i*stride : i*stride+seqlen]
        
    return seqs
    
'''

def indent(depth):
    return f''.join(['--' for i in range(depth-1)]) + '> '


def main_DELM():

    args = parse_args()
    
    logger = get_logger(args.verbose)
    logger.info('Started training with args:')
    logger.info('\n'.join([f'{indent(1)}{k}: {vars(args)[k]}' for k in vars(args)]))

    SEQLEN = 10
    TRAIN_SPLIT = 0.8
    TEST_SPLIT = 1.0 - TRAIN_SPLIT
    
    queries_path = 'preprocessing/arrays/small/queries/'
    
    if args.tf_data_api:
        # tf.data API approach <---
        train = tf.data.Dataset.from_generator(
                lambda: seq_generator_from_folder(
                    os.path.join(queries_path, 'train'), 
                    stride=args.stride, 
                    seqlen=args.seqlen, 
                    include_start=args.include_start),
                output_signature=tf.TensorSpec(shape=(10,2), dtype=tf.string))
        
        test = tf.data.Dataset.from_generator(
                lambda: seq_generator_from_folder(
                    os.path.join(queries_path, 'test'), 
                    stride=args.stride, 
                    seqlen=args.seqlen, 
                    include_start=args.include_start),
                output_signature=tf.TensorSpec(shape=(10,2), dtype=tf.string))
        
        train = train.batch(args.bs).prefetch(tf.data.AUTOTUNE)
        test = test.batch(args.bs).prefetch(tf.data.AUTOTUNE)
        # --->
    
    else:
        # Old approach <---
        seqs = create_sequences(os.path.join(queries_path, 'queries-20160509_011632.pcap.csv.npy'), SEQLEN)
        train, test = seqs[: round(len(seqs) * TRAIN_SPLIT)], seqs[round(len(seqs) * TRAIN_SPLIT) :]
        # --->
    
    model = build_model('delm', seqlen=args.seqlen)
    
    if args.load and len(os.listdir('checkpoints')) > 0:
        last_checkpoint = os.listdir('checkpoints')[[os.path.getmtime(os.path.join('checkpoints', f)) for f in os.listdir('checkpoints')].index(max([os.path.getmtime(os.path.join('checkpoints', f)) for f in os.listdir('checkpoints')]))]
        last_chk_path = os.path.join('checkpoints', last_checkpoint)
        
        logger.debug(f'Trying to load model weights from {last_chk_path}...')
        
        logger.info(f'Calling model to initialize layers...')
        model(list(train.take(1).unbatch().as_numpy_iterator())[0:1])
        
        model.load_weights(last_chk_path)
        logger.info(f'Model weights loaded from {last_chk_path}.')
    
    if args.debug:
        with open('preprocessing/vocabs/small/domains_vocab.txt', 'r') as f:
            domains_vocab = [l.strip() for l in f.readlines()]
        
        seq = train.unbatch().take(1).as_numpy_iterator()
        seq = np.array([s for s in seq])

        seq[0,0,1] = '<MASK>'
        seq[0,1,1] = '<MASK>'
        
        pred = model(seq)
        pred = tf.nn.softmax(pred)
        pred = pred[0]

        for i in range(len(pred)):
            logger.info(f'{seq[0,i,0]} {seq[0,i,1]} -> {domains_vocab[np.array(pred).argmax(axis=-1)[i]]} ({100*(np.array(pred).max(axis=-1)[i]):.2f}%)')
        
        sys.exit(0)
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    chk_path = f'checkpoints/model-{time.strftime("%y%m%d-%H%M%S", time.localtime())}.h5'
    
    logger.debug('Starting model training...')
    
    model.fit(
            x=train, 
            y=None if args.tf_data_api else train, 
            validation_data=test if args.tf_data_api else (test, test), 
            validation_freq=1, 
            batch_size=args.bs, 
            epochs=args.epochs, 
            callbacks=[ModelCheckpoint(chk_path, monitor='loss', save_weights_only=True)]
    )
    '''
    if args.tf_data_api:
        #for x in train.take(1):
        #    model.test_step(x)
        #    start_time = time.perf_counter()
        #    model.test_step(x)
        #print(f'{time.perf_counter() - start_time} seconds.')
        model.fit(x=train, validation_data=test, validation_freq=1, batch_size=args.bs, epochs=args.epochs, callbacks=[ModelCheckpoint(chk_path, monitor='loss', save_weights_only=True)])
    else:
        model.fit(x=train, y=train, validation_data=(test, test), validation_freq=1, batch_size=args.bs, epochs=args.epochs, callbacks=[
                          #TqdmCallback(data_size=len(train), batch_size=args.bs, verbose=2),
                          ModelCheckpoint(chk_path, monitor='loss', save_weights_only=True)])
    '''
    logger.debug(f'Model training completed.')
    
    model.save_weights(chk_path)
    
    logger.debug('Starting model evaluation...')
    model.evaluate(x=test, y=None if args.tf_data_api else test, batch_size=args.bs)
    logger.debug('Model evaluation completed.')
    
    
if __name__ == '__main__':
    main_DELM()
   
