import random
import os
import numpy as np
import torch
import os.path as osp
import yaml
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def seed_setting(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA 10.2+
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)

def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)
        
def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path, weights_only=False)
    return None



def extract_classes_subgraph(graph_data, target_classes=None, num_classes=6):
    """
    Extract a subgraph containing only specified classes.
    
    Args:
        graph_data: Original graph Data object
        target_classes: List of class indices to keep (e.g., [0, 1, 2, 3, 4, 5])
                       If None, uses first num_classes
        num_classes: Number of first classes to keep if target_classes is None
    
    Returns:
        New Data object with only the specified classes
    """
    # Determine which classes to keep
    if target_classes is None:
        # Get unique classes and sort them
        unique_classes = torch.unique(graph_data.y).sort()[0]
        # Take first num_classes
        target_classes = unique_classes[:num_classes].tolist()
    
    print(f"Extracting classes: {target_classes}")
    
    # Create mask for nodes belonging to target classes
    node_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    for class_idx in target_classes:
        node_mask |= (graph_data.y == class_idx)
    
    # Get indices of nodes to keep
    node_indices = torch.where(node_mask)[0]
    
    print(f"Original graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
    print(f"Keeping {node_indices.shape[0]} nodes from {len(target_classes)} classes")
    
    # Extract subgraph (edges between selected nodes)
    sub_edge_index, edge_mask = subgraph(
        subset=node_indices,
        edge_index=graph_data.edge_index,
        relabel_nodes=True,  # This will relabel nodes from 0 to N-1
        num_nodes=graph_data.num_nodes
    )
    
    # Extract node features and labels for selected nodes
    sub_x = graph_data.x[node_indices]
    sub_y = graph_data.y[node_indices]
    
    # Handle edge features if they exist
    sub_xe = None
    if hasattr(graph_data, 'xe') and graph_data.xe is not None:
        sub_xe = graph_data.xe[edge_mask]
    
    # Handle masks (train/val/test)
    sub_train_mask = None
    sub_val_mask = None
    sub_test_mask = None
    
    if hasattr(graph_data, 'train_mask') and graph_data.train_mask is not None:
        sub_train_mask = graph_data.train_mask[node_indices]
    
    if hasattr(graph_data, 'val_mask') and graph_data.val_mask is not None:
        sub_val_mask = graph_data.val_mask[node_indices]
    
    if hasattr(graph_data, 'test_mask') and graph_data.test_mask is not None:
        sub_test_mask = graph_data.test_mask[node_indices]
    
    # Create new Data object
    new_graph_data = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        y=sub_y,
        xe=sub_xe,
        train_mask=sub_train_mask,
        val_mask=sub_val_mask,
        test_mask=sub_test_mask
    )
    
    print(f"New graph: {new_graph_data.num_nodes} nodes, {new_graph_data.edge_index.shape[1]} edges")
    
    # Print class distribution
    print("\nClass distribution in new graph:")
    for class_idx in target_classes:
        count = (sub_y == class_idx).sum().item()
        print(f"  Class {class_idx}: {count} nodes")
    
    return new_graph_data


def extract_classes_preserving_structure(graph_data, target_classes=None, num_classes=6, 
                                        preserve_edges=True):
    """
    Alternative method that can optionally preserve edges to removed nodes.
    
    Args:
        graph_data: Original graph Data object
        target_classes: List of class indices to keep
        num_classes: Number of first classes to keep if target_classes is None
        preserve_edges: If True, keeps edges even if one endpoint is removed
                       (useful for maintaining graph connectivity)
    
    Returns:
        New Data object with modified structure
    """
    if target_classes is None:
        unique_classes = torch.unique(graph_data.y).sort()[0]
        target_classes = unique_classes[:num_classes].tolist()
    
    # Create mask for target nodes
    node_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
    for class_idx in target_classes:
        node_mask |= (graph_data.y == class_idx)
    
    if preserve_edges:
        # Keep all nodes but mark removed ones
        new_x = graph_data.x.clone()
        new_y = graph_data.y.clone()
        
        # Set labels of removed nodes to a special value (e.g., -1)
        new_y[~node_mask] = -1
        
        # Optionally zero out features of removed nodes
        new_x[~node_mask] = 0
        
        new_graph_data = Data(
            x=new_x,
            edge_index=graph_data.edge_index,
            y=new_y,
            xe=graph_data.xe if hasattr(graph_data, 'xe') else None,
            train_mask=graph_data.train_mask & node_mask if hasattr(graph_data, 'train_mask') else None,
            val_mask=graph_data.val_mask & node_mask if hasattr(graph_data, 'val_mask') else None,
            test_mask=graph_data.test_mask & node_mask if hasattr(graph_data, 'test_mask') else None
        )
    else:
        # Use the standard extraction method
        new_graph_data = extract_classes_subgraph(graph_data, target_classes, num_classes)
    
    return new_graph_data