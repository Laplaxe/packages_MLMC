import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#set device to "cuda" if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoregressiveMasking(object):
    """Autoregressive constraint for weight matrices."""
    def __init__(self, frequency=1):
        """
        Constructor for AutoregressiveMasking.

        Parameters:
        - frequency (int): Controls how often the autoregressive constraint is applied.
        """
        self.frequency = frequency

    def __call__(self, module):
        """
        Applies the autoregressive constraint to the weight matrices of a module.

        Parameters:
        - module: PyTorch module to which the constraint is applied.
        """
        # Apply autoregressive constraint to weight matrices
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = torch.tril(w, -1)  # Apply lower triangular masking
            module.weight.data = w

class local_made(nn.Module):
    """Local Autoregressive Made"""
    def __init__(self, border_size, inside_size, device = "cuda"):
        """
        Constructor for the local_made model.

        Parameters:
        - border_size (int): Size of the border in the input. It is the fixed region that does not get updated in MC
        - inside_size (int): Size of the inside region in the input. It is the region that gets updated in MC
        """
        super(local_made, self).__init__()
        self.border_size = border_size
        self.inside_size = inside_size

        self.inside_layer = nn.Linear(inside_size, inside_size, bias=False)
        self.border_layer = nn.Linear(border_size, inside_size, bias=False)
        self.activation = nn.Sigmoid()

        self.dev = device

    def forward(self, x):
        """
        Forward pass of the local_made model.

        Parameters:
        - x (torch.Tensor): Input tensor to the model.

        Returns:
        - torch.Tensor: Output tensor after the forward pass.
        """
        if x.dim() == 1:
            x1, x2 = torch.split(x, [self.border_size, self.inside_size], dim=0) #split data into border and inside region
        else:
            x1, x2 = torch.split(x, [self.border_size, self.inside_size], dim=1) #split data into border and inside region
        x1 = self.border_layer(x1) #propagate border
        x2 = self.inside_layer(x2) #propagate inside
        x = x1 + x2 #sum the effects
        x = self.activation(2 * x)
        if self.dev == "cuda":
            torch.cuda.empty_cache()  # Clear GPU cache
        return x


def train_local_made(totdata, border_size, inside_size, epochs=50, batch_size=256, learning_rate=1e-3, device = "cuda"):
    """
    Train the local_made architecture using data.

    Parameters:
    - data (torch.Tensor): Training data as a PyTorch tensor.
    - border_size (int): Size of the border in the input.
    - inside_size (int): Size of the inside region in the input.
    - epochs (int): Number of training epochs.
    - patience (int): patience of the model
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.
    - check: wheather to print or not the graphs
    
    Returns:
    - local_made: Trained local_made model.
    """
    data = torch.clone(totdata)
    model = local_made(border_size, inside_size, device = device)
    model = model.to(device)
    clipper = AutoregressiveMasking()
    model.train()
    model.inside_layer.apply(clipper) #apply autoregressive constraint only on inside layer

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")

    #train
    for epoch in tqdm(range(epochs)):
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            batch_data = data[indices]
            #batch_data = data[indices].to(device)
            _, batch_data_out = torch.split(batch_data, [border_size, inside_size], dim=1)

            # Forward pass
            output = model(batch_data)

            # Compute loss
            loss = criterion(output, (batch_data_out + 1) / 2)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.inside_layer.apply(clipper)

    return model

"""
def train_local_made(totdata, border_size, inside_size, epochs=50, patience = 20, batch_size=256, learning_rate=1e-3, device = "cuda", check = True):
    
    Train the local_made architecture using data.

    Parameters:
    - data (torch.Tensor): Training data as a PyTorch tensor.
    - border_size (int): Size of the border in the input.
    - inside_size (int): Size of the inside region in the input.
    - epochs (int): Number of training epochs.
    - patience (int): patience of the model
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.
    - check: wheather to print or not the graphs
    
    Returns:
    - local_made: Trained local_made model.
    
    data = totdata[:int(len(totdata)*0.9)]
    valdata = totdata[int(len(totdata)*0.9):]
    _, valdata_output = torch.split(valdata, [border_size, inside_size], dim=1)
    model = local_made(border_size, inside_size, device = device)
    model = model.to(device)
    bestmodel = local_made(border_size, inside_size, device = device)
    bestmodel = bestmodel.to(device)
    clipper = AutoregressiveMasking()
    model.train()
    model.inside_layer.apply(clipper) #apply autoregressive constraint only on inside layer

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_curve=[]
    val_curve=[]
    lr_curve=[]

    best_val_loss = 100000
    epochs_since_best_val_loss = 0
    best_epoch = epochs-1

    #train
    for epoch in tqdm(range(epochs)):
        model.train()
        tmp_loss = 0
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            batch_data = data[indices]
            #batch_data = data[indices].to(device)
            _, batch_data_out = torch.split(batch_data, [border_size, inside_size], dim=1)

            # Forward pass
            output = model(batch_data)

            # Compute loss
            loss = criterion(output, (batch_data_out + 1) / 2)
            tmp_loss += loss.detach().cpu().numpy()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.inside_layer.apply(clipper)

        if epoch > 50:
            scheduler.step()

        lr_curve.append(optimizer.param_groups[0]['lr'])
        train_curve.append(tmp_loss/len(data))
        # Validation
        model.eval() # the validation step does NOT change the parameters
        with torch.no_grad():
            output = model(valdata)
            val_loss = criterion(output, (valdata_output + 1) / 2).item()
            val_loss /= len(valdata)
        
            val_curve.append(val_loss)

            # Check if the validation accuracy has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_best_val_loss = 0
                bestmodel.load_state_dict(model.state_dict())
                best_epoch = epoch
            else:
                epochs_since_best_val_loss += 1

            # Check if early stopping is necessary
            if epochs_since_best_val_loss >= patience:
                break
    if check:
        f,ax = plt.subplots(1,1)
        ax2 = ax.twinx()
        ax.plot(train_curve,label="train",color="blue")
        ax.plot(val_curve,label="validation",color="orange")
        ax2.axvline(best_epoch,color="r",linestyle="--",label="best epoch")
        ax.legend(loc=1)
        ax.set_ylabel("Loss")

        ax2.plot(lr_curve,"k--",label="LR",)
        ax2.legend(loc=2)
        plt.xlabel("Epochs")
        ax2.set_ylabel("LR")
        ax.set_yscale("log")
        return bestmodel, f
    else:
        return bestmodel, None
"""

def train_existing_local_made(model, data, border_size, inside_size, epochs=50, patience = 20, batch_size=256, learning_rate=1e-2, device = "cuda"):
    """
    Train an existing local_made architecture using data.

    Parameters:
    - model (local_made): pre-existing made one wishes to train
    - data (torch.Tensor): Training data as a PyTorch tensor.
    - border_size (int): Size of the border in the input.
    - inside_size (int): Size of the inside region in the input.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for optimization.

    Returns:
    - local_made: Trained local_made model.
    """
    model = model.to(device)
    clipper = AutoregressiveMasking()
    model.train()
    model.inside_layer.apply(clipper) #apply autoregressive constraint only on inside layer

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction="sum")

    #train
    for epoch in tqdm(range(epochs)):
        model.train()
        for i in range(0, len(data), batch_size):
            indices = random.sample(range(data.shape[0]), batch_size)
            batch_data = data[indices].to(device)
            _, batch_data_out = torch.split(batch_data, [border_size, inside_size], dim=1)

            # Forward pass
            output = model(batch_data)

            # Compute loss
            loss = criterion(output, (batch_data_out + 1) / 2)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.inside_layer.apply(clipper)

    return model