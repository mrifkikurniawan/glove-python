# GloVe model from the NLP lab at Stanford:
# http://nlp.stanford.edu/projects/glove/.
import array
import collections
import io
import joblib
try:
    # Python 2 compat
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import scipy.sparse as sp
import numbers
import gc

from .glove_cython import fit_vectors, transform_paragraph


def check_random_state(seed):
    """ Turn seed into a np.random.RandomState instance.

        This is a copy of the check_random_state function in sklearn
        in order to avoid outside dependencies.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class Glove(object):
    """
    Class for estimating GloVe word embeddings using the
    corpus coocurrence matrix.
    """

    def __init__(self, dictionary: dict=None, no_components=30, learning_rate=0.05,
                 alpha=0.75, max_count=100, max_loss=10.0,
                 random_state=None,
                 pretrained: str=None):
        """
        Parameters:
        - int no_components: number of latent dimensions
        - float learning_rate: learning rate for SGD estimation.
        - float alpha, float max_count: parameters for the
          weighting function (see the paper).
        - float max_loss: the maximum absolute value of calculated
                          gradient for any single co-occurrence pair.
                          Only try setting to a lower value if you
                          are experiencing problems with numerical
                          stability.
        - random_state: random statue used to intialize optimization
        """
        
        if not dictionary == None:
            num_vocab = len(dictionary)
        self.no_components = no_components
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.max_count = float(max_count)
        self.max_loss = max_loss

        self.word_vectors = None
        self.word_biases = None

        self.vectors_sum_gradients = None
        self.biases_sum_gradients = None

        self.random_state = random_state
        
        # initialize weights
        random_state = check_random_state(self.random_state)
        
        if not dictionary == None:
            print("Initialize word vectors")
            self.word_vectors = ((random_state.rand(num_vocab,
                                                    self.no_components) - 0.5)
                                / self.no_components)

        if not dictionary == None:
            print("Initialize word bias")
            self.word_biases = np.zeros(num_vocab,
                                        dtype=np.float64)

        if not dictionary == None:        
            print("Initialize gradients")
            self.vectors_sum_gradients = np.ones_like(self.word_vectors)
            self.biases_sum_gradients = np.ones_like(self.word_biases)
        
        # dictionary, words_to_ids
        if not dictionary == None:
            print("add dictionary")
            self.__add_dictionary(dictionary)
        
        # load pre-trained model
        if not dictionary == None:
            if pretrained:
                print(f"Loading pretrained from {pretrained}")
                self.load_glove(pretrained)
            else:
                print(f"Train Glove model from scratch")
            

    def fit(self, matrix, epochs=5, no_threads=2, verbose=False):
        """
        Estimate the word embeddings.

        Parameters:
        - scipy.sparse.coo_matrix matrix: coocurrence matrix
        - int epochs: number of training epochs
        - int no_threads: number of training threads
        - bool verbose: print progress messages if True
        """

        shape = matrix.shape

        if (len(shape) != 2 or
            shape[0] != shape[1]):
            raise Exception('Coocurrence matrix must be square')

        if not sp.isspmatrix_coo(matrix):
            raise Exception('Coocurrence matrix must be in the COO format')
        
        random_state = check_random_state(self.random_state)
        shuffle_indices = np.arange(matrix.nnz, dtype=np.int32)

        if verbose:
            print('Performing %s training epochs '
                  'with %s threads' % (epochs, no_threads))

        for epoch in range(epochs):

            if verbose:
                print('Epoch %s' % epoch)

            # Shuffle the coocurrence matrix
            random_state.shuffle(shuffle_indices)

            fit_vectors(self.word_vectors,
                        self.vectors_sum_gradients,
                        self.word_biases,
                        self.biases_sum_gradients,
                        matrix.row,
                        matrix.col,
                        matrix.data,
                        shuffle_indices,
                        self.learning_rate,
                        self.max_count,
                        self.alpha,
                        self.max_loss,
                        int(no_threads))

            if not np.isfinite(self.word_vectors).all():
                raise Exception('Non-finite values in word vectors. '
                                'Try reducing the learning rate or the '
                                'max_loss parameter.')

    def transform_paragraph(self, paragraph, epochs=50, ignore_missing=False):
        """
        Transform an iterable of tokens into its vector representation
        (a paragraph vector).

        Experimental. This will return something close to a tf-idf
        weighted average of constituent token vectors by fitting
        rare words (with low word bias values) more closely.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit to transform paragraphs')

        if self.dictionary is None:
            raise Exception('Dictionary must be provided to '
                            'transform paragraphs')

        cooccurrence = collections.defaultdict(lambda: 0.0)

        for token in paragraph:
            try:
                cooccurrence[self.dictionary[token]] += self.max_count / 10.0
            except KeyError:
                if not ignore_missing:
                    raise

        random_state = check_random_state(self.random_state)

        word_ids = np.array(cooccurrence.keys(), dtype=np.int32)
        values = np.array(cooccurrence.values(), dtype=np.float64)
        shuffle_indices = np.arange(len(word_ids), dtype=np.int32)

        # Initialize the vector to mean of constituent word vectors
        paragraph_vector = np.mean(self.word_vectors[word_ids], axis=0)
        sum_gradients = np.ones_like(paragraph_vector)

        # Shuffle the coocurrence matrix
        random_state.shuffle(shuffle_indices)
        transform_paragraph(self.word_vectors,
                            self.word_biases,
                            paragraph_vector,
                            sum_gradients,
                            word_ids,
                            values,
                            shuffle_indices,
                            self.learning_rate,
                            self.max_count,
                            self.alpha,
                            epochs)

        return paragraph_vector

    def __add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()
        
        self.inverse_dictionary = {v: k for k, v in items_iterator}

    def save(self, filename):
        """
        Serialize model to filename.
        """
        import gc
        gc.collect()
        joblib.dump(self.__dict__,
                    filename)   
    
    def save_joblib(self, filename):
        import gc
        
        gc.collect()
        joblib.dump(self.__dict__,
                    filename)    

    @classmethod
    def load(cls, filename):
        """
        Load model from filename.
        """

        instance = Glove()
        instance.__dict__ = joblib.load(filename)
        instance.word_vectors = instance.__dict__['word_vectors']
        instance.dictionary = instance.__dict__['dictionary']
        instance.inverse_dictionary = instance.__dict__['inverse_dictionary']
        instance.no_components = instance.__dict__['no_components']
        instance.learning_rate = instance.__dict__['learning_rate'] 

        return instance

    @classmethod
    def load_stanford(cls, filename):
        """
        Load model from the output files generated by
        the C code from http://nlp.stanford.edu/projects/glove/.

        The entries of the word dictionary will be of type
        unicode in Python 2 and str in Python 3.
        """

        dct = {}
        vectors = array.array('d')

        # Read in the data.
        with io.open(filename, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.split(' ')

                word = tokens[0]
                entries = tokens[1:]

                dct[word] = i
                vectors.extend(float(x) for x in entries)

        # Infer word vectors dimensions.
        no_components = len(entries)
        no_vectors = len(dct)

        # Set up the model instance.
        instance = Glove()
        instance.no_components = no_components
        instance.word_vectors = (np.array(vectors)
                                 .reshape(no_vectors,
                                          no_components))
        instance.word_biases = np.zeros(no_vectors)
        instance.add_dictionary(dct)

        return instance

    def _similarity_query(self, word_vec, number):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]

    def most_similar(self, word, number=5):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(self.word_vectors[word_idx], number)[1:]

    def most_similar_paragraph(self, paragraph, number=5, **kwargs):
        """
        Return words most similar to a given paragraph (iterable of tokens).
        """

        paragraph_vector = self.transform_paragraph(paragraph, **kwargs)

        return self._similarity_query(paragraph_vector, number)


    def load_glove(self, pretrained_path: str) -> dict:
        print("Updating old vectors with pre-trained vectors")
        count_replacements = 0
        count_not_replacements = 0
        with open(pretrained_path, 'rb') as file:
            for line in file:
                line = line.decode().split()
                word = line[0]
                if word in self.dictionary:
                    vec = line[1:]
                    len_vec = len(vec)
                    if len_vec > self.no_components:
                        dif = len_vec - self.no_components
                        vect = np.array(vec[dif:]).astype(np.float)    
                    else:
                        vect = np.array(vec).astype(np.float)
                    
                    idx = self.dictionary[word]
                    self.word_vectors[idx, :] = vect
                    count_replacements+=1
                else:
                    count_not_replacements+=1
        
        print(f"Vocab with pretrained: {count_replacements}")
        print(f"Vocab with random: {count_not_replacements}")