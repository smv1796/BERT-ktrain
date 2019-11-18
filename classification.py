# import ktrain
import ktrain
from ktrain import text

# download IMDb movie review dataset
import tensorflow as tf
'''dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz", 
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
    extract=True,
)

# set path to dataset
import os.path
#dataset = '/root/.keras/datasets/aclImdb'
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
'''
IMDB_DATADIR='C:/Users/MTREE/.keras/datasets/aclImdb'
print(IMDB_DATADIR)

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(IMDB_DATADIR,
                                                                       maxlen=64,
                                                                       preprocess_mode='bert',
                                                                       train_test_names=['train',
                                                                                         'test'],
                                                                       classes=['pos', 'neg'])

model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model,train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=2)

learner.fit_onecycle(2e-5, 1)