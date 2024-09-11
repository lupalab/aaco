import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import yaml
from itertools import chain, combinations
from xgboost import XGBClassifier
import os
import sys
sys.path.append('./src')
from classifier import classifier_xgb_dict, classifier_ground_truth, classifier_xgb, NaiveBayes
from mask_generator import random_mask_generator, all_mask_generator



# Load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load data based on the dataset provided in the configuration
def load_data(dataset_name):
    if dataset_name == "cube":
        ### Read in Cube data
        cube = np.load("input_data/cube_20_0.3.pkl", allow_pickle=True)
        cube_train = cube.get('train')
        X_train = torch.from_numpy(cube_train[0])
        y_train = F.one_hot(torch.from_numpy(cube_train[1]).long())
        cube_val = cube.get('valid')
        X_valid = torch.from_numpy(cube_val[0])
        y_valid = F.one_hot(torch.from_numpy(cube_val[1]).long())
        initial_feature = 6

    elif dataset_name == "grid":
        ### Read in the grid data
        grid = np.load("input_data/grid_data.pkl", allow_pickle=True)
        grid_train = grid.get('train')
        X_train = grid_train[0]
        y_train = F.one_hot(grid_train[1].squeeze().long())
        grid_val = grid.get('valid')
        X_valid = grid_val[0]
        y_valid = F.one_hot(grid_val[1].squeeze().long())
        ### Normalize data
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 1

    elif dataset_name == "gas10":
        ### Read in the gas data
        gas = np.load("input_data/gas.pkl", allow_pickle=True)
        gas_train = gas.get('train')
        X_train = torch.from_numpy(gas_train[0])
        y_train = F.one_hot(torch.from_numpy(gas_train[1]).long())
        gas_val = gas.get('valid')
        X_valid = torch.from_numpy(gas_val[0])
        y_valid = F.one_hot(torch.from_numpy(gas_val[1]).long())
        # Normalize features
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 6

    elif dataset_name == "MNIST":
        ### Read in the MNIST data
        mnist = np.load("input_data/MNIST.pkl", allow_pickle=True)
        mnist_train = mnist.get('train')
        X_train = mnist_train[0]
        y_train = mnist_train[1]
        mnist_val = mnist.get('valid')
        X_valid = mnist_val[0]
        y_valid = mnist_val[1]
        # Normalize features
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 100

    feature_count = X_train.shape[1]  # Total number of features in the dataset
    return X_train, y_train, X_valid, y_valid, initial_feature, feature_count


# Load the appropriate classifier based on dataset and model
def load_classifier(dataset_name, X_train, y_train, input_dim):
    if dataset_name == "cube":
        # Use the ground truth classifier for Cube dataset
        return classifier_ground_truth(num_features=20, num_classes=8, std=0.3)
    
    elif dataset_name == "grid" or dataset_name == "gas10":
        # Use XGB dictionary classifier for Grid and Gas10 datasets
        return classifier_xgb_dict(output_dim=y_train.shape[1], input_dim=input_dim, subsample_ratio=0.01, X_train=X_train, y_train=y_train)

    elif dataset_name == "MNIST":
        # Load XGBoost model for MNIST dataset
        xgb_model = XGBClassifier()
        xgb_model.load_model('models/xgb_classifier_MNIST_random_subsets_5.json')
        return classifier_xgb(xgb_model)

    else:
        raise ValueError("Unsupported dataset or model")

        

def get_knn(X_train, X_query, masks, num_neighbors, instance_idx=0, exclude_instance=True):
    """
    Args:
    X_train: N x d Train Instances
    X_query: 1 x d Query Instances
    masks: d x R binary masks to try
    num_neighbors: Number of neighbors (k)
    """
    X_train_squared = X_train ** 2
    X_query_squared = X_query ** 2
    X_train_X_query = X_train * X_query
    dist_squared = torch.matmul(X_train_squared, masks) - 2.0 * torch.matmul(X_train_X_query, masks) + torch.matmul(X_query_squared, masks)
    
    if exclude_instance:
        idx_topk = torch.topk(dist_squared, num_neighbors + 1, dim=0, largest=False)[1]
        return idx_topk[idx_topk != instance_idx][:num_neighbors]
    else:
        return torch.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]


# Helper function to load the mask generator based on the dataset
def load_mask_generator(dataset_name, input_dim):
    if dataset_name in ["cube", "MNIST"]:
        return random_mask_generator(10000, input_dim, 1000)
    elif dataset_name == "grid" or dataset_name == "gas10":
        all_masks = generate_all_masks(input_dim)  # Generate all possible masks for grid and gas10
        return all_mask_generator(all_masks)
    else:
        raise ValueError("Unsupported dataset for mask generation")
        
        
        
def aaco_rollout(X_train, y_train, X_valid, y_valid, classifier, mask_generator, initial_feature, config):
    # Load parameters from the config
    feature_count = X_train.shape[1]
    acquisition_cost = config['acquisition_cost']
    nearest_neighbors = config['nearest_neighbors']
    hide_val = 10
    num_instances = config['num_instances']  # Number of instances to loop through
    
    # Decide whether to use training or validation data
    if config['train_or_validation'] == "train":
        X = X_train
        y = y_train
        not_i = True  # Ensure instance isn't its own neighbor in KNN
    else:
        X = X_valid
        y = y_valid
        not_i = False  # Allow instance to be its own neighbor in KNN
    
    # Initialize lists to store results
    X_rollout = []
    y_rollout = []
    action_rollout = []
    mask_rollout = []
    
    # Define the loss function
    loss_function = nn.CrossEntropyLoss(reduction='none')

    ##############################################
    ##### AACO Rollout
    ##############################################
    
    for i in range(num_instances):  # Loop through the specified number of instances
        print(f"Starting instance {i} at {datetime.datetime.now()}")
        
        # Initialize the current mask (start with no features)
        mask_curr = torch.zeros((1, feature_count))
        
        for j in range(feature_count + 1):
            if j == 0:
                # Select the initial feature deterministically
                mask_rollout.append(mask_curr.clone().detach())
                mask_curr[0, initial_feature] = 1
                action = torch.zeros(1, feature_count + 1)
                action[0, initial_feature] = 1
                X_rollout.append(X[[i]])
                y_rollout.append(y[[i]])
                action_rollout.append(action)
            else:
                # Get the nearest neighbors based on the observed feature mask
                idx_nn = get_knn(X_train, X[[i]], mask_curr.T, nearest_neighbors, i, not_i).squeeze()
                print(f"Neighbors gathered for instance {i} at {datetime.datetime.now()}")
                
                # Generate random masks and get the next set of possible masks
                new_masks = mask_generator(mask_curr)
                mask = torch.maximum(new_masks, mask_curr.repeat(new_masks.shape[0], 1))
                mask[0] = mask_curr  # Ensure the current mask is included
                
                # Get only unique masks
                mask = mask.unique(dim=0)
                n_masks_updated = mask.shape[0]
                
                # Predictions based on the classifier
                x_rep = X_train[idx_nn].repeat(n_masks_updated, 1)
                mask_rep = torch.repeat_interleave(mask, nearest_neighbors, 0)
                idx_nn_rep = idx_nn.repeat(n_masks_updated)
                y_pred = classifier(torch.cat([torch.mul(x_rep, mask_rep) - (1 - mask_rep) * hide_val, mask_rep], -1), idx_nn)
                
                # Compute loss
                loss = loss_function(y_pred, y_train[idx_nn].repeat(n_masks_updated, 1).float()) + acquisition_cost * mask_rep.sum(dim=1)
                loss = torch.stack([loss[i * nearest_neighbors:(i+1) * nearest_neighbors].mean() for i in range(n_masks_updated)])
                
                # Find the best mask (one with the lowest loss)
                loss_argmin = loss.argmin()
                mask_i = mask[loss_argmin]
                mask_diff = mask_i - mask_curr
                
                # Check if no new features are acquired
                if mask_diff.sum().item() == 0:
                    # No more features to acquire, add prediction action
                    action = torch.zeros(1, feature_count + 1)
                    action[0, feature_count] = 1  # Action to predict (last column indicates prediction)
                    action_rollout.append(action)
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    mask_rollout.append(mask_curr)
                    break
                else:
                    # If new features are acquired, choose the feature with the lowest expected loss
                    non_zero = mask_diff.nonzero()[:, 1]
                    ordering_masks = mask_curr.repeat(len(non_zero), 1)
                    ordering_masks[range(len(non_zero)), non_zero] = 1
                    ordering_masks = ordering_masks.repeat_interleave(nearest_neighbors, 0)
                    
                    x_ordering = X_train[idx_nn].repeat(len(non_zero), 1)
                    idx_nn_ordering = idx_nn.repeat(len(non_zero))
                    y_pred = classifier(torch.cat([torch.mul(x_ordering, ordering_masks) - (1 - ordering_masks) * hide_val, ordering_masks], -1), idx_nn)
                    
                    # Compute loss for feature acquisition
                    loss = loss_function(y_pred, y_train[idx_nn].repeat(len(non_zero), 1).float())
                    avg_loss = torch.stack([loss[i * nearest_neighbors:(i + 1) * nearest_neighbors].mean() for i in range(len(non_zero))])
                    
                    action_idx = non_zero[avg_loss.argmin()]
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    mask_rollout.append(mask_curr.clone().detach())
                    action = torch.zeros(1, feature_count + 1)
                    action[0, action_idx] = 1
                    action_rollout.append(action)
                    
                    # Update the current mask
                    mask_curr[:, action_idx] = 1
    
    # Save the results
    results_dir = './results/'
    os.makedirs(results_dir, exist_ok=True)
    
    data = {
        'X': torch.cat(X_rollout), 
        'mask': torch.cat(mask_rollout), 
        'Action': torch.cat(action_rollout), 
        'y': torch.cat(y_rollout)
    }
    
    file_name = f"{results_dir}dataset_{config['dataset']}_rollout.pt"
    torch.save(data, file_name)
    print(f"Results saved to {file_name}")

        
if __name__ == "__main__":
    # Load configuration file
    config = load_config("config.yaml")

    # Load data based on dataset name in config
    X_train, y_train, X_valid, y_valid, initial_feature, feature_count = load_data(config['dataset'])

    # Extract other parameters from configuration
    nearest_neighbors = config['nearest_neighbors']
    acquisition_cost = config['acquisition_cost']
    train_or_validation = config['train_or_validation']
    dataset_name = config['dataset']
    
    print("Data loaded successfully")
    print("Timestamp:", datetime.datetime.now())
    
    # Load the classifier based on the dataset and outcome model
    classifier = load_classifier(dataset_name, X_train, y_train, input_dim=X_train.shape[1])

    print("Classifier loaded")
    print("Timestamp:", datetime.datetime.now())
    
    # Load the appropriate mask generator based on dataset
    mask_generator = load_mask_generator(dataset_name, feature_count)

    print("Masks generated")
    print("Timestamp:", datetime.datetime.now())
    
    aaco_rollout(X_train, y_train, X_valid, y_valid, classifier, mask_generator, initial_feature, config)
    


    
