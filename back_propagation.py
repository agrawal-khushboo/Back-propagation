import numpy as np
import pickle
import matplotlib.pyplot as plt
from os import listdir
config = {}
config['layer_specs'] = [784,47,47,
                         10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 500  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.001  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001  # Learning rate of gradient descent algorithm


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    e = np.exp(x - np.amax(x,axis=1,keepdims=True))
    output = e/np.sum(e,axis=1,keepdims=True)
    return output


def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    """
    # Get the list of image file names
    file = open('./'+fname, 'rb')
    input = pickle.load(file)
    # Store the images as arrays and their labels in two lists
    images = input[:,:784]
    labels  = input[:,784]
    label_coded = []
    for label in labels:
        l = [0,0,0,0,0,0,0,0,0,0]
        l[int(label)] = 1
        label_coded.append(l)
    return images, label_coded


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None  # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.

    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = 1.0/(1.0+np.exp(-x))
        return output

    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = np.tanh(x)
        return output

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = np.max(0,x)
        return output

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        s = self.sigmoid(self.x)
        grad = s*(1-s)
        return grad

    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        grad = 1 - np.square(np.tanh(self.x))
        return grad

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.where(self.x > 0 ,1,0)
        return grad


class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.v_w = np.zeros((in_units, out_units))
        self.v_b = np.zeros((1, out_units)).astype(np.float32)
        self.w = np.random.randn(in_units, out_units)   # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        self.a = np.dot(x,self.w)+self.b
        return self.a

    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        lam = config['L2_penalty']
        N = self.x.shape[0]
        self.d_w = np.dot(self.x.T,delta)+lam/N*self.w
        self.d_b = np.mean(delta,axis=0)
        self.d_x = np.dot(delta,self.w.T)
        return self.d_x


class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets
        for layer in self.layers:
            x = layer.forward_pass(x)

        self.y = softmax(x)
        loss = None
        if targets != None:
            loss = self.loss_func(self.y,targets)
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        epsilon = 1e-9
        lam = config['L2_penalty']
        N = logits.shape[0]
        loss = 0
        logits = np.matrix(logits)
        logists = logits + epsilon
        logists = np.log(logists)
        targets = np.matrix(targets)
        loss = np.multiply(targets, logists)
        loss = loss.sum()

        for i in range(len(self.layers)):
            if i % 2 == 0:
                loss += lam / (2 * N) * np.sum(np.square(self.layers[i].w))
        return -loss / logits.shape[0]
        # epsilon = 1e-9
        # lam = config['L2_penalty']
        # N = logits.shape[0]
        # loss = 0
        #
        # #loss = np.mean(targets*np.log(logits+epsilon))
        # for i in range(logits.shape[0]):
        #     for j in range(10):
        #         loss += targets[i][j] * np.log(logits[i][j]+epsilon)
        # for i in range(len(self.layers)):
        #     if i%2 ==0:
        #         loss += lam/(2*N)*np.sum(np.square(self.layers[i].w))
        # return -loss/logits.shape[0]

    def backward_pass(self):
        '''
        implement the backward pass for the whole network.
        hint - use previously built functions.
        '''
        delta = self.y - self.targets
        i = len(self.layers) - 1
        delta =  self.y - self.targets
        while i >= 0:
            delta = self.layers[i].backward_pass(delta)
            i = i - 1


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
    Write the code to train the network. Use values from config to set parameters
    such as L2 penalty, number of epochs, momentum, etc.
    """
    epochs = config['epochs']
    mu = config['momentum_gamma']
    learning_rate = config['learning_rate']
    layers = model.layers
    epoch_set = []
    Training_loss_set = []
    Validation_loss_set = []
    Training_accuracy_set = []
    validation_accuracy_set = []
    epoch_set.append(0)
    training_loss, train = model.forward_pass(X_train, y_train)
    Training_loss_set.append(training_loss)
    training_accuracy = test(model, X_train, y_train,config)
    Training_accuracy_set.append(training_accuracy)
    validation_loss,valid = model.forward_pass(X_valid, y_valid)
    Validation_loss_set.append(validation_loss)
    validation_accuracy = test(model, X_valid, y_valid,config)
    validation_accuracy_set.append(validation_accuracy)
    weights = {}
    bias = {}
    v_w = {}
    v_b = {}
    data = X_valid
    for i in range(len(layers)):
        if i % 2 == 0:
            weights[i] = layers[i].w
            bias[i] = layers[i].b
            v_w[i] = layers[i].v_w
            v_b[i] = layers[i].v_b
    if config['momentum'] == True:
        mu = config['momentum_gamma']
    else:
        mu=0
    increasing_loss = 0
    stop_at = config['early_stop_epoch']
    for epoch in range(epochs):
        print(validation_loss)
        epoch_set.append(epoch+1)
        batches = int(X_train.shape[0]/config['batch_size'])
        for b in range(batches):
            start = b*config['batch_size']
            end = (b+1)*config['batch_size']
            X = X_train[start:end]
            Y = y_train[start:end]
            model.forward_pass(X, Y)
            model.backward_pass()
            for i in range(len(layers)):
                if i % 2 == 0:
                    layers[i].v_w = mu * layers[i].v_w - learning_rate * layers[i].d_w
                    layers[i].w += layers[i].v_w
                    layers[i].d_b = mu * layers[i].d_b - learning_rate * layers[i].d_b
                    layers[i].b += layers[i].d_b
        training_loss, train = model.forward_pass(X_train, y_train)
        Training_loss_set.append(training_loss)
        training_accuracy = test(model, X_train, y_train,config)
        Training_accuracy_set.append(training_accuracy)
        new_validation_loss,valid = model.forward_pass(X_valid, y_valid)
        Validation_loss_set.append(validation_loss)
        validation_accuracy = test(model, X_valid, y_valid,config)
        validation_accuracy_set.append(validation_accuracy)
        if config['early_stop'] == True and new_validation_loss >= validation_loss:
            increasing_loss+=1
            if increasing_loss == stop_at:
                layers[i].v_w = v_w[i]
                layers[i].v_b = v_b[i]
                layers[i].w = weights[i]
                layers[i].b = bias[i]
                break;
        else:
            increasing_loss = 0
            validation_loss = new_validation_loss
            v_w[i] = layers[i].v_w
            v_b[i] = layers[i].v_b
            weights[i] = layers[i].w
            bias[i] = layers[i].b
    plt.plot(epoch_set, Training_loss_set, 'r', label='Training set loss')
    plt.plot(epoch_set, Validation_loss_set, 'b', label='Validation set loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss For tanh')
    plt.show()
    plt.plot(epoch_set, Training_accuracy_set, 'r', label='Training set Accuracy')
    plt.plot(epoch_set, validation_accuracy_set, 'b', label='Validation set Accuracy')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy For tanh')
    plt.show()

def test(model, X_test, y_test, config):
    """
    Write code to run the model on the data passed as input and return accuracy.
    """
    loss, y_pred = model.forward_pass(X_test)
    correct = []
    for i in range(X_test.shape[0]):
        argpre = np.argmax(y_pred[i])
        argtest = np.argmax(y_test[i])
        if argpre == argtest:
            correct.append(1)
        else:
            correct.append(0)

    accuracy =  sum(correct)/len(correct)
    return accuracy


if __name__ == "__main__":
    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    trainer(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test, config)
    print('Accuracy',test_acc)


