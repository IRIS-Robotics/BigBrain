import numpy as np

from src.tokenizer import TOKENIZE
from src.one_hot import OneHot
from src.ngrams import Ngram

class SkipGram:

    def __init__(self, window_size, embdding_size, alpha, corpus_path):
        '''
        :param window_size: the size of the window
        :param tokenize_matrix: the tokenize matrix of the text
        !param corpus_size: the size of the corpus
        :param embdding_size: the size of the embedding
        :param alpha: the learning rate
        '''

        self.window_size = window_size
        self.embdding_size = embdding_size
        self.alpha = alpha

        with open(corpus_path, "r", encoding="utf8") as f:
            self.corpus = f.read()

        self.tokenize = TOKENIZE(self.corpus)
        self.tokenize_matrix, self.filtred_corpus = self.tokenize.show_sentences()

        # remove the duplicates
        self.filtred_corpus = list(dict.fromkeys(self.filtred_corpus))

        self.corpus_size = len(self.filtred_corpus)
        print(f"Corpus size: {self.corpus_size}")


        # random initialization of the weights
        self.output = np.zeros((self.corpus_size, self.embdding_size))
        self.weights = np.random.rand(self.corpus_size, self.embdding_size)
        self.biases = np.zeros(self.corpus_size)

        self.onehot_dict = None

    def train(self, epochs):

        one_hot = OneHot(self.filtred_corpus)
        encoded_words, self.onehot_dict = one_hot.encode()
        for epoch in range(epochs):
            print("==================================================")
            print(f"IRIS NLP Training {epoch + 1}/{epochs} epochs")
            print("==================================================")
            self.forward_propagation()



    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def crossentropy_loss(self, softmax_output, target_vec):
        # Compute the cross-entropy loss
        loss = -np.sum(target_vec * np.log(softmax_output.T))

        # Compute the gradient of the cross-entropy loss with respect to the softmax output
        grad = softmax_output.T - target_vec

        # Compute the gradient of the loss with respect to the weights
        grad_weights = np.dot(target_vec.T, grad)
        grad_biases = np.mean(grad, axis=1)

        # Update the weights and biases
        self.weights -= self.alpha * grad_weights
        self.biases -= self.alpha * grad_biases

        return loss, grad

    def forward_propagation(self):

        ngram = Ngram(self.filtred_corpus, self.window_size)
        ngram.context()
        # for each target word in the corpus
        for i, target_word in enumerate(self.filtred_corpus):
            # skip the target word if it doesn't have any context
            if len(self.onehot_dict[target_word]) == 0:
                continue

            context_words = ngram.get_context_words(i)
            # print(f"Target word: {target_word} | Context words: {context_words}")
            # iterate over the context words for the target word
            for context_word in context_words:

                # encoded context
                encoded_word = np.array(self.onehot_dict[context_word])
                encoded_word = encoded_word.reshape(encoded_word.shape[0], 1)
                output = np.dot(self.weights.T, encoded_word) + self.biases
                softmax_output = self.softmax(output)

                # print("Softmax output: ", softmax_output)
                # print("Encoded word: ", encoded_word)

                loss, softmax_grad = self.crossentropy_loss(softmax_output, encoded_word)
                print(f"Loss: {loss}")


    def predict(self, word):
        # compute the output
        encoded_word = np.array(self.onehot_dict[word])
        encoded_word = encoded_word.reshape(encoded_word.shape[0], 1)
        output = np.dot(self.weights.T, encoded_word) + self.biases
        # compute the softmax
        softmax_output = self.softmax(output)

        # get the index of the max value
        index = np.argmax(softmax_output, axis=1)
        # get key into list of keys
        print(f"Index: {index}")
        print(f"Word: {self.filtred_corpus[index[0]]}")

        return self.filtred_corpus[index[0]]


    def get_tokenized_matrix(self):
        return self.tokenize_matrix

    def get_filtred_corpus(self):
        return self.filtred_corpus

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases



if __name__ == '__main__':

    skipgram = SkipGram(window_size=5, embdding_size=5, alpha=0.01, corpus_path="../assets/corpus/asian/asian.txt")
    # print(skipgram.get_tokenized_matrix())
    # print(skipgram.get_filtred_corpus())
    # print(skipgram.get_weights())
    # print(skipgram.get_biases())

    skipgram.train(5)
    # print(skipgram.get_weights())
    # print(skipgram.get_biases())
    skipgram.predict("femme")



