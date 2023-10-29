import numpy as np
import sys
import json
from scipy.optimize import minimize
from copy import deepcopy as copy

global opt_hist
opt_hist = []

def main():
    #Predefined from generated QAOA instances
    N = 9 #9 qubit QAOA circuit
    p = 3 #3 layer QAOA circuit

    #Pass argument for number of NN layers and number of neurons per layer
    no_layers = int(sys.argv[1])
    nodes_per_layer = []
    for layer in range(no_layers):
        nodes_per_layer += [ int(sys.argv[layer + 2]) ] 
    regularisation = float(sys.argv[no_layers+2])
    test_backprop = bool(int(sys.argv[no_layers+3]))
    
    no_features = int(N*(N-1)/2) # Simply take features as the Ising coefficients of which there are N(N-1)/2 many.
    no_outputs = 2*p # Want to output to be the 2*p QAOA angles

    #Initialize network
    network = initialize_network(no_layers=no_layers, nodes_per_layer=nodes_per_layer, rng_seed=39124)
    
    #If argument is passed to test backpropagation for random inputs not from the training set.
    #This will do 10 tests for each regularisation parameters listed below, with 10 random inputs.
    #Compares the approximated gradient via derivative formula against gradient via backpropagated error.
    if test_backprop:
        for test_reg in [0.5, 1.0, 100.0]:
            print("Regularisation: ", test_reg)
            check_backpropagate(network, no_features, no_outputs, test_reg)
   
    #Training of network begins here
    global opt_hist
    opt_hist = []

    #10 randomly initialised networks
    networks = [network]
    for netidx in range(9):
        network = initialize_network(no_layers=no_layers, nodes_per_layer=nodes_per_layer, rng_seed=39124+netidx)
        networks += [network]

    best_network_err = float('inf')
    x_inputs = np.loadtxt("datasets/training_x_inputs.csv", delimiter=",")
    y_outputs = np.loadtxt("datasets/training_y_outputs.csv", delimiter=",")
    for network in networks:
    #Train each, keeping only the one with lowest error
        _opt_result, _opt_hist = train_network(
            network,
            nodes_per_layer, 
            x_inputs, 
            y_outputs, 
            regularisation=regularisation
            )
        if _opt_result.fun < best_network_err:
            best_network_err = _opt_result.fun
            opt_result = _opt_result
            opt_hist = copy(_opt_hist)   
    n_iters = int(opt_result.nit)
    opt_weights = list(opt_result.x)
    opt_loss = float(opt_result.fun)
    training_data = [
        opt_weights,
        opt_loss,
        opt_hist,
        n_iters
        ]
    print("Final training Error: ", opt_result.fun)
    savefile = "trained_networks/NN[{}]_reg={}.json".format(",".join([str(x) for x in nodes_per_layer]), regularisation)
    print("Saving training data to ", savefile)
    with open(savefile, "w") as f:
        json.dump(training_data, f, indent=2)

    return None

def initialize_network(no_layers=1, nodes_per_layer=[], rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    network = []
    for k in range(no_layers - 1):
        network += [ rng.random( size = (nodes_per_layer[k+1],nodes_per_layer[k]+1) ) ] #+1 for bias
    return network

def activate_layer(weights, inputs):
    activation = np.matmul( weights, inputs )
    return activation

def add_bias_terms(x):
    n, m = x.shape #n = no_features, m = no_samples
    biased_x = np.ones(shape=(n+1, m)) #Extra feature/node for bias in 0th place
    biased_x[1:,:] = x 
    return biased_x

def backpropagated_gradient(network, x, y, regularisation=0.0):
    no_layers = len(network) + 1
    network_delta = [ None ]*(no_layers)
    output, a_s = forward_pass(network, x, return_intermediate=True)  #Note: output should equal a_s[no_layers-1] so it is not stored in a_s to save memory
    for i in reversed(range(1,no_layers)):
        if i == no_layers-1:
            delta = (output - y) * output * (1 - output)
            network_delta[i] = delta
        else:
            weights = network[i]
            delta_right = network_delta[i+1] if i==no_layers-2 else network_delta[i+1][1:,:] #If not using delta from output, don't include bias delta in position 0 of 1st dimension
            output = a_s[i]
            delta = ( np.matmul(weights.T, delta_right) ) * output * (1 - output)
            network_delta[i] = delta
    
    _, m = x.shape #m = no_samples
    gradient = [ np.zeros_like(weights) for weights in network ]
    
    for i in range(no_layers-1):
        for k in range(m):
            delta_right = network_delta[i+1][:,k] if i==no_layers-2 else network_delta[i+1][1:,k]
            output = a_s[i][:,k]
            gradient[i] += np.outer(delta_right, output)
            
        #Regularisation
        gradient[i][:,1:] += regularisation * network[i][:,1:]

        gradient[i] /= m

    return gradient

def flatten_weights(weights_shaped):
    weights_flat = []
    for _weights in weights_shaped:
        weights_flat = np.append( weights_flat, _weights.flatten() )
    return weights_flat

def reshape_weights(weights_flat, nodes_per_layer):
    weights = []
    no_params_reshaped = 0
    no_layers = len(nodes_per_layer)
    for k in range(no_layers - 1):
        weights += [ weights_flat[no_params_reshaped: no_params_reshaped + nodes_per_layer[k+1]*(nodes_per_layer[k]+1) ].reshape( (nodes_per_layer[k+1],nodes_per_layer[k]+1) ) ]
        no_params_reshaped += nodes_per_layer[k+1]*(nodes_per_layer[k]+1)
    return weights

def forward_pass(network, x, return_intermediate = False):
    no_layers = len(network) + 1
    new_inputs = add_bias_terms(x) #Add bias terms before activating
    activations = [new_inputs] #First activation layer is input layer
    for k in range(no_layers-1):
        z = activate_layer(network[k], new_inputs)
        new_inputs = g(z)
        #Don't add bias at output layer
        if k != no_layers - 2:
            new_inputs = add_bias_terms(new_inputs)
            if return_intermediate:
                activations += [new_inputs]
    if return_intermediate:
        return new_inputs, activations
    else:
        return new_inputs

def cost(network, x, y, regularisation=0.0):
    m = x.shape[1]
    #Regularise each weight matrix except bias weights
    reg_cost = 0
    for wt in network:
        reg_cost += np.sum( regularisation * wt[:,1:] * wt[:,1:] ) 
    delta = y - forward_pass(network, x)
    return ( np.sum( delta * delta ) + reg_cost ) / (2*m)

def approx_costgrad(network, x, y, idx, regularisation=0.0):
    eps = 1e-4
    weights = flatten_weights(network)
    weights_plus = weights.copy()
    weights_plus[idx] += eps
    weights[idx] -= eps
    nodes_per_layer = [network[0].shape[1]-1]
    for weight_mat in network:
        nodes_per_layer += [ weight_mat.shape[0] ]
    network_plus = reshape_weights(weights_plus,nodes_per_layer)
    network = reshape_weights(weights,nodes_per_layer)
    return ( cost(network_plus, x, y, regularisation=0.0) - cost(network, x, y, regularisation=0.0) ) / ( 2 * eps )

#Check for 10 random inputs and outputs up to 10 times (total 5 checks), that gradients are close:
def check_backpropagate(network, no_features, no_outputs, regularisation=0.0, no_samples=10, no_trials=5, rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    nodes_per_layer = [network[0].shape[1]-1]
    for weight_mat in network:
        nodes_per_layer += [ weight_mat.shape[0] ]
    for t in range(no_trials):
        x = rng.random( size=(no_features, no_samples) )
        y = rng.random( size=(no_outputs, no_samples) )
        gradient = backpropagated_gradient(network, x, y, regularisation=regularisation)
        gradient = flatten_weights(gradient)
        approx_gradient = np.zeros_like(gradient)
        
        #Note approx_gradient is not regularised
        for i in range(len(gradient)):
            approx_gradient[i] = approx_costgrad(network, x, y, idx=i, regularisation=regularisation) 
        
        #To regularise approx_gradient
        approx_grad_arrs = reshape_weights(approx_gradient, nodes_per_layer)
        for i, grad in enumerate(approx_grad_arrs):
            grad[:,1:] += regularisation * network[i][:,1:] / no_samples #Regularise each weight matrix except bias weights

        #Flatten regularised approx_gradient
        approx_gradient = flatten_weights( approx_gradient )
        print("Trial {} Backpropagated and approximated gradients agree: ".format(t+1), np.allclose(gradient, approx_gradient, 1e-2))

def train_network(network, nodes_per_layer, x_inputs, y_outputs, regularisation=0.0):
    x0 = flatten_weights(network)
    x0 = x0.astype(np.float128)
    global opt_hist
    opt_hist = []
    def callback(xk):
        global opt_hist
        opt_hist += [ list(xk) ]
    def cost_fun(weights, nodes_per_layer, inputs, outputs):
        network = reshape_weights(weights, nodes_per_layer)
        return cost(network, x=inputs, y=outputs)
    def backprop_grad(weights, nodes_per_layer, inputs, outputs):
        network = reshape_weights(weights, nodes_per_layer)
        gradient = backpropagated_gradient(network, x=inputs, y=outputs, regularisation=regularisation)
        return flatten_weights(gradient)
    opt_result = minimize(
        fun=cost_fun, 
        x0=x0, 
        jac=backprop_grad,
        method='L-BFGS-B',
        args=(nodes_per_layer, x_inputs, y_outputs), 
        callback=callback
        )
    return opt_result, copy( opt_hist )

#Sigmoid function g(z) = 1/(1+exp(-z))
def g(z):
    c = z.astype(dtype=np.float128)
    return 1/(1+np.exp(-c))

if __name__ == "__main__":
    main()