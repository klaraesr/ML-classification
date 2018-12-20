import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class NN(torch.nn.Module):
    """
    Your Neural Netowrk model.
    """
    dim_in = 2
    size_hidden = 12
    dim_out = 1
    num_epochs = 50
    size_batch = 5
    learning_rate = 0.01

    def __init__(self):
        super(NN, self).__init__()
        self.in_lay = torch.nn.Linear(self.dim_in, self.size_hidden)
        self.hidden = torch.nn.ReLU()
        self.out_lay = torch.nn.Linear(self.size_hidden, self.dim_out)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, X):
        """
        X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
        :return: Output of your neural network model. A pytorch Tensor of dimensions [number of samples].
        
        """
        to_hid = self.in_lay(X.float())
        hid = self.hidden(to_hid)
        out = self.out_lay(hid)
        out_sig = self.sigmoid(out)
        #out = self.layer(X.float())
        return out_sig


def train(model, X, y):
    """
    model: Neural network model. An instance of the NN class.
    X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
    y: Targets. A pytorch Tensor of dimensions [number of samples].
    :return: Nothing.
    """
    tensor_set = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset=tensor_set, batch_size=100,shuffle=True)
    
    # Loss function
    criterion = torch.nn.BCELoss()
    # Stochastic Gradient Descent optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(150):
        for i, (features, labels) in enumerate(train_loader):
            features = torch.autograd.Variable(X)
            labels = torch.autograd.Variable(y).float()
            #features = X.float()
            #labels = y.float()
            m = torch.nn.Sigmoid()
            y_pred = model(features)
            # print(labels)
            # print(y_pred)
            loss = criterion(y_pred, labels) # Calculate loss
            print('epoch: ', epoch,' loss: ', loss.item())
            optimizer.zero_grad() # Set gradients to zero before backpropagation
            loss.backward() # Backpropagation
            optimizer.step() # Update weights


def test(model, X):
    """
    model: Neural network model. An instance of the NN class.
    X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
    :return: Predicted targets. A pytorch Tensor of dimensions [number of samples].
    """

    features = torch.autograd.Variable(X)
    outputs = model(X)
    #_, predicted = torch.max(outputs.data, 1) # Choose the best class from the output: The class with the best score
    print(outputs)

def to_binary(tensor):
    if tensor < 0.5:
        return 0
    else:
        return 1
