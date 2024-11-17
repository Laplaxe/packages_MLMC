"""Definitions necessary for the global steps"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import random
from utilities import compute_energy

#set device to "cuda" if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_config_local_made(model, border, N_spins, N_config):
    """
    Generate N_config of N_spins using the local autoregressive model.

    Parameters:
    - model: Local autoregressive model used for spin generation.
    - border: Tensor representing the border configuration.
    - N_spins: Number of spins in each configuration.
    - N_config: Number of configurations to generate.

    Returns:
    - Tensor: A tensor containing generated spin configurations with shape (N_config, N_spins).
    """
    with torch.no_grad():
        # Initialize a tensor with random binary configurations (values of -1 or 1)
        config = (torch.bernoulli(torch.full((N_config, N_spins), 0.5)) * 2 - 1).to(device)
        
        # Generate each spin in an autoregressive manner
        for n in range(N_spins):
            # Get probabilities from the local autoregressive model for the nth spin
            probs = model(torch.cat((border, config), 1))
            
            # Sample new spin values based on probabilities and update the configuration
            config[:, n] = (torch.bernoulli(probs[:, n]) * 2 - 1)
            torch.cuda.empty_cache()
        
        return config

def generate_local_made(model, border, current_config, new_beta, N, J, N_data=100000, N_configs=10):
    """
    Perform Metropolis-Hastings updates on spin configurations using the local MADE.

    Parameters:
    - model: Local autoregressive model used for spin generation.
    - border: Tensor representing the border configuration.
    - current_config: Tensor representing the current spin configuration.
    - new_beta: Inverse temperature for the Metropolis-Hastings update.
    - N: Number of spins.
    - J: Interaction matrix.
    - N_data: Number of data points to generate in each iteration (default is 100000).
    - N_configs: Number of configurations to generate in each iteration (default is 10).

    Returns:
    - Tuple: A tuple containing two elements.
        - Tensor: Updated spin configurations after Metropolis-Hastings updates with shape (N_config, N_spins).
        - float: Acceptance rate of proposed configurations.
    """
    with torch.no_grad():
        # Initialize acceptance rate counter
        acc_rate = 0
        
        # Binary Cross Entropy Loss function
        bce = nn.BCELoss(reduction="none")
        
        # Store the current configurations
        old_configs = current_config
        torch.cuda.empty_cache()
        
        # Perform Metropolis-Hastings updates
        for t in range(N_configs):
            # Generate new configurations using the local autoregressive model
            new_configs = generate_config_local_made(model, border, N, N_data)
            
            # Calculate energies and arguments for old and new configurations
            energy_old = compute_energy(torch.cat((border, old_configs), 1), J, take_mean=False)
            arg_old = -new_beta * energy_old + torch.sum(bce(model(torch.cat((border, old_configs), 1)), (old_configs + 1) / 2), axis=1)
            
            new_energies = compute_energy(torch.cat((border, new_configs), 1), J, take_mean=False)
            temp = torch.sum(bce(model(torch.cat((border, new_configs), 1)), (new_configs + 1) / 2), axis=1)
            arg_new = -new_beta * new_energies + temp
            
            # Acceptance probability calculation
            acc = (torch.log(torch.rand(size=(N_data,))).to(device) < (arg_new - arg_old)).int()
            acc_rate += torch.sum(acc)
            
            # Update configurations based on acceptance
            old_configs = torch.einsum("i, ij->ij", (1 - acc), old_configs) + torch.einsum("i, ij->ij", acc, new_configs)
            
            torch.cuda.empty_cache()
        
        # Calculate and return the acceptance rate
        return old_configs, float(acc_rate / N_data / N_configs)
