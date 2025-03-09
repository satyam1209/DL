import numpy as np
class Neuron:
    def __init__(self,name):
        self.name = name
        self.forward=[]
        self.backward=[]
        self.weights=[]
        self.bias= np.random.uniform(-1, 1)
        self.value = 0
        self.delta = 0
def assign_weights(n):
    if n.weights==[]:
        if n.backward==[]:
            n.weights=[np.random.rand()]
        else:
            n.weights= [np.random.uniform(-1, 1) for _ in range(len(n.backward))]
    # if n.bias==[]:
    #     if n.backward==[]:
    #         n.bias=[np.random.rand()]
    #     else:
    #         n.bias=[np.random.rand(1,10) for i in range(len(n.backward))]
            
def Create_Neural_Network(no_of_layers,no_of_neurons_each_layer):
    #decide the number of layers and number of neurons in each layer
    neural_list = []
    for i in range(no_of_layers):
        layer = []
        for j in range(no_of_neurons_each_layer[i]):
            if i==0:
                n=Neuron(str("I")+str(j+1))
            elif i==no_of_layers-1:
                n=Neuron(str("O")+str(j+1))  
            else:
                n=Neuron(str("HL")+str(i)+str(j+1))
            layer.append(n)
        neural_list.append(layer)
    #assign weights to the neurons and create forward and backward connections
    for i in range(1,len(neural_list)):
        for j in neural_list[i]:
            for k in neural_list[i-1]:
                j.backward.append(k)
                k.forward.append(j)
    for i in range(len(neural_list)):
        for j in neural_list[i]:
            assign_weights(j)
    
    return neural_list
    #printing the forward and backward connections of each neuron
    # for i in neural_list:
    #     for j in i:
    #         print(j.name)
    #         print('Forward:',[k.name for k in j.forward])
    #         print('Backward:',[k.name for k in j.backward])
    #         print('Weights:',j.weights)
    #         print('\n')
def sigmoid_derivative(x):
    return x * (1 - x) + 1e-7

def forward_propagation(neural_list):
    for i in range(1,len(neural_list)):
        for j in neural_list[i]:
            j.value = sum(j.weights[k] * j.backward[k].value for k in range(len(j.backward)))
            j.value+=j.bias
            j.value = 1 / (1 + np.exp(-j.value))  # Apply sigmoid activation

    # return the output of the last neuron
    return neural_list[-1][0].value

def backpropagation(neural_list, target, learning_rate=0.1):
    output_layer = neural_list[-1]

    for i, neuron in enumerate(output_layer):
        neuron.delta = neuron.delta = (neuron.value - target[i]) * sigmoid_derivative(neuron.value)


    for i in range(len(neural_list) - 2, 0, -1):  # Hidden layers
        for neuron in neural_list[i]:
            # neuron.delta = sum(next_neuron.weights[neuron.forward.index(next_neuron)] * next_neuron.delta for next_neuron in neuron.forward)
            neuron.delta = sum(next_neuron.weights[j] * next_neuron.delta for j, next_neuron in enumerate(neuron.forward))

            neuron.delta *= sigmoid_derivative(neuron.value)  

    for i in range(1, len(neural_list)):
        for neuron in neural_list[i]:
            for j, prev_neuron in enumerate(neuron.backward):  
                neuron.weights[j] -= learning_rate * neuron.delta * prev_neuron.value
            neuron.bias -= learning_rate * neuron.delta  
 
                
def compute_loss(y_pred, y_true):
    return 0.5 * (y_true - y_pred) ** 2 

def train(neural_list, X, Y, epochs=100000, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X, Y):
            # Set input values
            for i in range(len(neural_list[0])):
                neural_list[0][i].value = x[i]

            # Forward pass
            output = forward_propagation(neural_list)

            # Compute loss
            total_loss += compute_loss(output, y)  # Assuming 1 output neuron

            # Backpropagation
            backpropagation(neural_list, [y], learning_rate)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
    return neural_list

def print_neural_network(neural_list):
    print("\nNeural Network Structure and Weights\n" + "="*50)
    
    for layer_idx, layer in enumerate(neural_list):
        layer_type = "Input Layer" if layer_idx == 0 else "Output Layer" if layer_idx == len(neural_list) - 1 else f"Hidden Layer {layer_idx}"
        print(f"\n{layer_type}:\n" + "-"*30)
        
        for neuron in layer:
            print(f"Neuron: {neuron.name}")
            print(f"  Bias: {neuron.bias:.4f}")
            if neuron.backward:  # If it has previous connections
                print(f"  Connections:")
                for i, prev_neuron in enumerate(neuron.backward):
                    print(f"    {prev_neuron.name} --> {neuron.name} (Weight: {neuron.weights[i]:.4f})")
            print()  # Empty line for readability



    

layers = 3
neurons_per_layer = [2,5,1]  



# Create network
neural_network = Create_Neural_Network(layers, neurons_per_layer)

# Training Data (XOR problem)

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_train = [1, 0, 0, 1]  # XOR outputs
# X_train = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9]]
# Y_train = [3, 5, 7, 9, 11, 13, 15, 17]
# print_neural_network(neural_network)
# Train the network
neural_network= train(neural_network, X_train, Y_train, epochs=50000, learning_rate=0.1)
# print_neural_network(neural_network)

for x in X_train:
    for i in range(len(neural_network[0])):
        neural_network[0][i].value = x[i]
    
    output = forward_propagation(neural_network)
    print(f"Input: {x}, Predicted Output: {output}")
        
    
        

    


        


        


        


