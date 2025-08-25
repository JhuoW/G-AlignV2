import torch
from typing import Optional
from torch_geometric.data import Data, Batch
from collections import defaultdict
from icl.icl import PrototypeInContextLearner
from utils.logging import logger
from data_process.data import KGDataset
from utils.utils import load_yaml
import os.path as osp
from data_process.datahelper import SentenceEncoder
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
import torch.nn.functional as F
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

    def load_kg_dataset(self, dataset_name):
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
        
        cache_dir = osp.join(kg_dataset.root, 'kg_pca_cache')
        os.makedirs(cache_dir, exist_ok=True)

        target_dim = self.cfg.unify_dim if hasattr(self.cfg, 'unify_dim') else 64

        data = kg_dataset.data
        node_feats = data.node_text_feat # (n_nodes, 768)
        relation_embeddings = data.class_node_text_feat

        graph_data = Data(x = data.x, edge_index = data.edge_index)  # graph data with just structure

        edge_types = data.edge_types      # (n_edges, )
        num_edge_types = kg_dataset.data.edge_types.max() + 1  # for fb15k237, num_edge_types = 237

        # load and compute pca-transformed node features
        node_feat_cache = osp.join(cache_dir, f'pca_{target_dim}.pt')
        if osp.exists(node_feat_cache):
            logger.info(f"Loading cached PCA node features from {node_feat_cache}")
            node_feats = torch.load(node_feat_cache, map_location='cpu')
        else:
            logger.info(f"Computing PCA node features (768 -> {target_dim})")
            node_feats = self.dimension_align(node_feats, cache_dir, dataset_name, 'node', target_dim)
            torch.save(node_feats.cpu(), node_feat_cache)
        

        relation_emb_cache = osp.join(cache_dir, f'pca_relation_{target_dim}.pt')
        if osp.exists(relation_emb_cache):
            relation_embeddings = torch.load(relation_emb_cache, map_location='cpu')
        else:
            relation_embeddings = self.dimension_align(relation_embeddings, cache_dir, dataset_name, 'relation', target_dim)
            torch.save(relation_embeddings.cpu(), relation_emb_cache)

        metadata = {'edge_types': edge_types,
                     'num_relations': int(edge_types.max().item()) + 1,
                     'relation_embeddings': relation_embeddings,
                     'dataset_name': dataset_name,
                     'num_nodes': node_feats.shape[0],
                     'num_edges': edge_types.shape[0],
                     'cache_dir': cache_dir}
        return graph_data, node_feats, metadata, kg_dataset


    def dimension_align(self, node_feats, cache_dir, dataset_name, feature_type, target_dim):
        target_dim = self.cfg.unify_dim if hasattr(self.cfg, 'unify_dim') else 64
        current_dim = node_feats.shape[1]
        if current_dim == target_dim:
            return node_feats
        
        pca_model_path = osp.join(cache_dir, f'pca_{feature_type}_{current_dim}_{target_dim}.pkl')

        if osp.exists(pca_model_path):
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            feat_np = node_feats.cpu().numpy()
            pca = PCA(n_components=target_dim, random_state=self.args.seed)
            pca.fit(feat_np)
            with open(pca_model_path, 'wb') as f:
                pickle.dump(pca, f)
        
        feat_np = feat_np.cpu().numpy()
        pca_x = pca.transform(feat_np)
        return torch.from_numpy(pca_x).float().to(node_feats.device)

    def _apply_pca_line_graph(self, features, dataset_name, kg_dataset):
        target_dim = self.cfg.unify_dim if hasattr(self.cfg, 'unify_dim') else 64
        current_dim = features.shape[1]
        
        if current_dim == target_dim:
            return features
        
        cache_dir = osp.join(kg_dataset.root, 'kg_pca_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        line_graph_features_cache = osp.join(cache_dir, f'line_graph_features_{current_dim}_{target_dim}.pt')
        pca_model_path = osp.join(cache_dir, f'pca_line_graph_{current_dim}_{target_dim}.pkl')
        
        if osp.exists(line_graph_features_cache):
            logger.info(f"Loading cached line graph features from {line_graph_features_cache}")
            cached_features = torch.load(line_graph_features_cache, map_location='cpu')
            # Verify shape matches
            if cached_features.shape == (features.shape[0], target_dim):
                return cached_features.to(features.device)
            else:
                logger.warning(f"Cached features shape mismatch, recomputing PCA")
        
        if osp.exists(pca_model_path):
            with open(pca_model_path, 'rb') as f:
                pca = pickle.load(f)
        else:
            features_np = features.cpu().numpy()
            pca = PCA(n_components=target_dim, random_state=42)
            pca.fit(features_np)
            
            with open(pca_model_path, 'wb') as f:
                pickle.dump(pca, f)
            logger.info(f"Saved PCA model to {pca_model_path}")
        
        features_np = features.cpu().numpy()
        features_reduced = pca.transform(features_np)
        features_tensor = torch.from_numpy(features_reduced).float()
        
        torch.save(features_tensor.cpu(), line_graph_features_cache)
        
        return features_tensor.to(features.device)

    @torch.no_grad()
    def link_classification_episode(self, graph_data, node_feats, metadata, k_shot, m_way, seed, kg_dataset):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = self.device
        graph_data = graph_data.to(device)
        node_feats = node_feats.to(device)
        edge_types = metadata['edge_types'].to(device)
        relation_embeddings = metadata['relation_embeddings'].to(device)
        num_relations = metadata['num_relations']

        if m_way is not None and m_way < num_relations:
            selected_relations = torch.randperm(num_relations)[:m_way].to(device)
            mask = torch.zeros_like(edge_types, dtype=torch.bool)
            for rel in selected_relations:
                mask |= (edge_types == rel)            
            edge_indices_to_keep = mask.nonzero(as_tuple=False).squeeze(-1)

            if edge_indices_to_keep.numel() == 0:
                logger.warning("No edges found for selected relations")
                return {'error': 'No edges for selected relations'}
            sub_edge_index = graph_data.edge_index[:, edge_indices_to_keep]
            sub_edge_types = edge_types[edge_indices_to_keep]            
            type_mapping = {rel.item(): i for i, rel in enumerate(selected_relations)}
            remapped_types = sub_edge_types.clone()
            for old_type, new_type in type_mapping.items():
                remapped_types[sub_edge_types == old_type] = new_type
            sub_edge_types = remapped_types
            sub_relation_embeddings = relation_embeddings[selected_relations]
        else:
            sub_edge_index = graph_data.edge_index
            sub_edge_types = edge_types
            sub_relation_embeddings = relation_embeddings
            m_way = num_relations
        sub_graph = Data(x=graph_data.x, edge_index=sub_edge_index)

        line_graph = self.line_transformer.transform(
            sub_graph,
            node_feats,
            sub_edge_types,
            sub_relation_embeddings
        )
        line_graph.x = self._apply_pca_line_graph(line_graph.x, metadata['dataset_name'], kg_dataset)
        line_graph = line_graph.to(device)
        if not hasattr(line_graph, 'batch'):
            line_graph.batch = torch.zeros(line_graph.x.shape[0], dtype=torch.long, device=device)

        num_line_nodes = line_graph.x.shape[0]
        support_mask = torch.zeros(num_line_nodes, dtype=torch.bool, device=device)
        for rel_type in range(m_way):
            rel_mask = (line_graph.y == rel_type)
            rel_indices = rel_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if rel_indices.numel() > 0:
                # Reserve at least half for query
                max_support = min(k_shot, len(rel_indices) // 2)
                if max_support > 0:
                    perm = torch.randperm(len(rel_indices))[:max_support]
                    selected = rel_indices[perm]
                    support_mask[selected] = True

        query_mask = ~support_mask

        if support_mask.sum() == 0 or query_mask.sum() == 0:
            logger.warning("Insufficient samples for support/query split")
            return {'error': 'Insufficient samples'}
        
        logger.info(f"Support: {support_mask.sum().item()} edges, Query: {query_mask.sum().item()} edges")

        domain_embedding = self.compute_domain_embedding(line_graph)
        
        gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
        gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
        gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
        H, _ = self.backbone_gnn.encode(
            line_graph.x,
            line_graph.edge_index,
            None,
            line_graph.batch
        )

        z_all = gamma_f * H + beta_f
        z_all = F.normalize(z_all, p=2, dim=-1)
        
        support_features = z_all[support_mask]
        support_labels = line_graph.y[support_mask]
        unique_labels = torch.unique(support_labels).sort()[0]
        
        prototypes = self.compute_prototypes(support_features, support_labels, unique_labels)        
        query_features = z_all[query_mask]
        query_labels = line_graph.y[query_mask]        
        similarities = self.prototype_distance(query_features, prototypes, 'cosine')
        predictions = similarities.argmax(dim=1)
        predicted_labels = unique_labels[predictions]
        accuracy = (predicted_labels == query_labels).float().mean().item()
        per_relation_acc = {}
        for rel in unique_labels:
            rel_mask = query_labels == rel
            if rel_mask.sum() > 0:
                rel_acc = (predicted_labels[rel_mask] == query_labels[rel_mask]).float().mean().item()
                per_relation_acc[rel.item()] = rel_acc        
        probs = F.softmax(similarities / 0.1, dim=1)
        confidences = probs.max(dim=1)[0]
        mean_confidence = confidences.mean().item()
        results = {
            'accuracy': accuracy,
            'per_relation_accuracy': per_relation_acc,
            'mean_confidence': mean_confidence,
            'num_support': support_mask.sum().item(),
            'num_query': query_mask.sum().item(),
            'num_relations': m_way,
            'k_shot': k_shot
        }
        return results
    

    def evaluate_link_classification(self, dataset_name, k_shot, m_way, n_episodes):
        graph_data, node_features, metadata, kg_dataset = self.load_kg_dataset(dataset_name)
        if m_way:
            logger.info(f"Using {m_way}-way classification")
        
        all_accuracies = []
        all_confidences = []
        all_per_relation = defaultdict(list)

        for episode in range(n_episodes):
            logger.info(f"\nEpisode {episode + 1}/{n_episodes}")
            results = self.link_classification_episode(graph_data,
                                                       node_features,
                                                       metadata,
                                                       k_shot=k_shot,
                                                       m_way=m_way,
                                                       seed=42 + episode)
            if 'error' not in results:
                all_accuracies.append(results['accuracy'])
                all_confidences.append(results['mean_confidence'])
                
                for rel, acc in results['per_relation_accuracy'].items():
                    all_per_relation[rel].append(acc)
        if not all_accuracies:
            return {'error': 'All episodes failed'}
        
        mean_per_relation = {}
        for rel, accs in all_per_relation.items():
            mean_per_relation[rel] = {'mean':np.mean(accs), 'std':np.std(accs)}
        
        aggregated_results = {'dataset': dataset_name,
                              'mean_accuracy': np.mean(all_accuracies),
                              'std_accuracy': np.std(all_accuracies),
                              'mean_confidence': np.mean(all_confidences),
                              'std_confidence': np.std(all_confidences),
                              'per_relation_accuracy': mean_per_relation,
                              'all_accuracies': all_accuracies,
                              'k_shot': k_shot,
                              'm_way': m_way if m_way else metadata['num_relations'],
                              'n_episodes': n_episodes}
        return aggregated_results


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

    learner = KGLinkClassificationICL(args, cfg)

    logger.info(f"\nDataset: {args.dataset}")
    logger.info(f"Settings: {args.k_shot}-shot, {args.m_way or 'all'}-way")
    logger.info(f"Episodes: {args.n_episodes}")

    results = learner.evaluate_link_classification(dataset_name=args.dataset,
                                                   k_shot=args.k_shot,
                                                   m_way=args.m_way,
                                                   n_episodes=args.n_episodes)
    if 'error' not in results:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Dataset: {results['dataset']}")
        print(f"Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"Confidence: {results['mean_confidence']:.4f} ± {results['std_confidence']:.4f}")
        print(f"Settings: {results['k_shot']}-shot, {results['m_way']}-way")
        print(f"Episodes: {results['n_episodes']}")        



if __name__ == "__main__":
    main()