import torch
from typing import Optional
from torch_geometric.data import Data, Batch
from collections import defaultdict
from icl.icl import PrototypeInContextLearner
from utils.logging import logger
from data_process.data import KGDataset
from utils.utils import load_yaml
import os.path as osp
from data_process.datahelper import SentenceEncoder, refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs
import numpy as np
import hydra
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)  # /home/zhuowei/Code/G-Align
from omegaconf import OmegaConf
from omegaconf import DictConfig
import argparse
from sklearn.decomposition import PCA
import os
import pickle

# print(data_config_lookup['data_config']['fb15k237'])
# {'task_level': 'e2e_link', 'args': {'remove_edge': True, 'walk_length': None}, 'dataset_splitter': 'KGSplitter', 'preprocess': 'KGConstructEdgeList', 'construct': 'ConstructKG', 'process_label_func': 'process_int_label', 'eval_metric': 'acc', 'eval_func': 'classification_func', 'eval_mode': 'max', 'dataset_name': 'fb15k237', 'num_classes': 237}

class LineGraphTransformer:
    def __init__(self, aggregate_method: str = 'concat', use_relation_emb: bool = True):
        """
        Args:
            aggregate_method: How to aggregate head and tail entity features
                - 'concat': Concatenate features [h; t] or [h; r; t]
                - 'mean': Average features (h + t)/2
                - 'hadamard': Element-wise product h * t
                - 'diff': Difference h - t
            use_relation_emb: Whether to include relation embeddings in edge features
        """
        self.aggregate_method = aggregate_method
        self.use_relation_emb = use_relation_emb

    def _create_edge_node_features(self, node_features, edge_index, edge_types, relation_embeddings):
        """Create features for edge-nodes in line graph."""
        head_idx, tail_idx = edge_index
        
        # Get features of head and tail entities
        head_features = node_features[head_idx]
        tail_features = node_features[tail_idx]
        
        # Aggregate entity features
        if self.aggregate_method == 'concat':
            if self.use_relation_emb and relation_embeddings is not None:
                # Get relation embeddings for each edge
                rel_features = relation_embeddings[edge_types]
                # Concatenate [head; relation; tail]
                edge_features = torch.cat([head_features, rel_features, tail_features], dim=-1)
            else:
                # Concatenate [head; tail]
                edge_features = torch.cat([head_features, tail_features], dim=-1)
        elif self.aggregate_method == 'mean':
            edge_features = (head_features + tail_features) / 2
            if self.use_relation_emb and relation_embeddings is not None:
                rel_features = relation_embeddings[edge_types]
                edge_features = (edge_features + rel_features) / 2
        elif self.aggregate_method == 'hadamard':
            edge_features = head_features * tail_features
            if self.use_relation_emb and relation_embeddings is not None:
                rel_features = relation_embeddings[edge_types]
                edge_features = edge_features * rel_features
        elif self.aggregate_method == 'diff':
            edge_features = head_features - tail_features
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate_method}")
        
        return edge_features
    
    def _create_line_graph_edges(self, edge_index):
        """
        Create edges for line graph.
        Two edges are connected if they share a head or tail entity.
        """
        num_edges = edge_index.shape[1]
        device = edge_index.device
        
        # Build adjacency lists for finding edges that share entities
        entity_to_edges = defaultdict(list)
        for edge_id in range(num_edges):
            head, tail = edge_index[0, edge_id].item(), edge_index[1, edge_id].item()
            entity_to_edges[head].append(edge_id)
            entity_to_edges[tail].append(edge_id)
        
        # Create edges in line graph
        line_edges = set()
        for entity_id, edge_list in entity_to_edges.items():
            # Connect all pairs of edges that share this entity
            for i in range(len(edge_list)):
                for j in range(i + 1, len(edge_list)):
                    line_edges.add((edge_list[i], edge_list[j]))
                    line_edges.add((edge_list[j], edge_list[i]))  # Bidirectional
        
        if line_edges:
            line_edges_list = list(line_edges)
            line_graph_edges = torch.tensor(line_edges_list, device=device).t()
        else:
            line_graph_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return line_graph_edges

    def transform(self, data, node_features, edge_types, relation_embeddings):
        """
        Transform KG to line graph representation.
        
        Args:
            data: Original KG with edge_index
            node_features: Node feature matrix (n_nodes, d)
            edge_types: Relation type for each edge (n_edges,)
            relation_embeddings: Optional relation embeddings (n_relations, d)
            
        Returns:
            line_graph: Transformed graph where edges become nodes
        """
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        device = edge_index.device
        
        # Create node features for line graph (each edge becomes a node)
        edge_features = self._create_edge_node_features(
            node_features, edge_index, edge_types, relation_embeddings
        )
        
        # Create edges in line graph (connect edges that share entities)
        line_graph_edges = self._create_line_graph_edges(edge_index)
        
        # Create line graph data object
        line_graph = Data(x = edge_features, edge_index = line_graph_edges, y = edge_types, num_nodes = num_edges)
        
        return line_graph
    

class KGLinkClassificationICL(PrototypeInContextLearner):
    def __init__(self, args, cfg):
        super().__init__(args)
        self.args = args
        self.cfg = cfg
        self.line_transformer = LineGraphTransformer(aggregate_method = args.aggr_method, use_relation_emb = args.use_relation_emb)
        self.pca_cache = {}

    def load_kg_dataset(self, dataset_name, llm_encoder):
        logger.info(f"Loading KG dataset {dataset_name} ...")
        # data_storage: datasets/
        llm_encoder = SentenceEncoder(self.args.llm_name, batch_size=self.cfg.llm_b_size)
        # Data(x=[14541, 1], 
        #      edge_index=[2, 310116],   # 边数
        #      edge_types=[310116],      # 边label
        #      node_text_feat=[14541, 768],  # 节点文本
        #      edge_text_feat=[1, 768],      
        #      class_node_text_feat=[237, 768])  kg_dataset.data
        # kg_dataset.root = datasets/ofa/KnowledgeGraph.FB15K237
        kg_dataset = KGDataset(cfg=self.cfg,  # 
                               name=dataset_name,
                               llm_encoder=llm_encoder,
                               load_text=True)
        
        data = kg_dataset.data
        node_feats = data.node_text_feat # (n_nodes, 768)
        relation_embeddings = data.class_node_text_feat

        edge_types = data.edge_types      
        num_edge_types = kg_dataset.data.edge_types.max() + 1  # for fb15k237, num_edge_types = 237



    def dimension_align(self, features, ds_name, feature_type):
        target_dim = self.cfg.unify_dim if hasattr(self.cfg, 'unify_dim') else 64
        current_dim = features.shape[1]
        if current_dim == target_dim:
            return features
        
        cache_key = f"{ds_name}_{feature_type}_{current_dim}_{target_dim}"
        if cache_key not in self.pca_cache:
            logger.info(f"Computing PCA for {feature_type} features: {current_dim} -> {target_dim}")
            features_np = features.cpu().numpy()
            pca = PCA(n_components=target_dim)
            pca_x = pca.fit_transform(features_np)

            self.pca_cache[cache_key] = pca
            cache_dir = osp.join(self.cfg.dirs.data_storage, 'kg_pca_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = osp.join(cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(pca, f)
            return torch.from_numpy(pca_x).float().to(features.device)
        else:
            pca = self.pca_cache[cache_key]
            features_np = features.cpu().numpy()
            pca_x = pca.transform(features_np)
            return torch.from_numpy(pca_x).float().to(features.device)

                

    # def prepare_line_graph(self, graph_data, edge_labels):
    #     line_graph = self.line_transformer.transform(graph_data, edge_labels)

    #     if self.line_transformer.aggregate_node_features == 'concat':
    #         original_dim = graph_data.x.size(1)
    #         line_dim = line_graph.x.shape[1]
    #         if line_dim != self.cfg.unify_dim:
    #             if self.feature_projector is None:





@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg:DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='generated_files/output/G-Align/final_gfm_model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--dataset', type=str, default='fb15k237',
                       choices=['fb15k237', 'wn18rr'],
                       help='KG dataset for link classification')
    parser.add_argument('--aggr_method', type=str, default='concat')
    parser.add_argument('--use_relation_emb', type=bool, default=True,
                       help='Whether to use relation embeddings')

    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per relation')
    parser.add_argument('--m_way', type=int, default=None,
                       help='Number of relations to classify (None = all)')

    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--llm_name', type=str, default='roberta',
                       choices=['ST', 'llama2_7b', 'llama2_13b', 'e5', 'roberta'],
                       help='Pretrained language model to use')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("="*60)
    logger.info("In-Context Link Classification")
    logger.info("="*60)
    OmegaConf.set_struct(cfg, False)
    # print(cfg.dirs.data_storage)
    llm_encoder = SentenceEncoder(args.llm_name, batch_size = cfg.llm_b_size)
    # Data(x=[14541, 1], 
    #      edge_index=[2, 310116],   # 边数
    #      edge_types=[310116],      # 边label
    #      node_text_feat=[14541, 768],  # 节点文本
    #      edge_text_feat=[1, 768],      
    #      class_node_text_feat=[237, 768])  kg_dataset.data
    
    kg_dataset = KGDataset(cfg=cfg,  # 
                            name=args.dataset,
                            llm_encoder=llm_encoder,
                            load_text=True)

if __name__ == "__main__":
    main()