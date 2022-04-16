import numpy as np

class NeuralNetwork:
    def __init__(self, layers, hyperparams=None):
        self.layers = layers
        self.weight_shapes = [(a,b) for a,b in zip(layers[1:], layers[:-1])]
        
        if hyperparams == None:
            self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in self.weight_shapes]
            self.biases = [np.zeros(s) for s in layers[1:]]
        else:
            self.set_flattened_hyperparams(hyperparams[0], hyperparams[1])
    
    def predict(self, feed):
        for w,b in zip(self.weights, self.biases):
            feed = np.tanh(np.matmul(w, feed) + b)      
        return feed
    
    def get_flattened_hyperparams(self):
        flat_weights = []
        flat_biases = []

        for dense_weights, bias_layer in zip(self.weights, self.biases):
            flat_weights = np.concatenate((flat_weights, dense_weights.flatten()))
            flat_biases = np.concatenate((flat_biases, bias_layer))
        
        return list(flat_weights), list(flat_biases)

    def set_flattened_hyperparams(self, weights, biases):
        self.weights = []
        self.biases = []

        bias_index = 0
        weight_index = 0
        for layer, weight_shape in zip(self.layers[1:], self.weight_shapes):
            self.weights.append(np.reshape(weights[weight_index:weight_index + np.prod(weight_shape)], weight_shape))
            self.biases.append(biases[bias_index:bias_index + layer])
            bias_index += layer
            weight_index += np.prod(weight_shape)
