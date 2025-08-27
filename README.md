# G-Align

```
datasets/
├── ofa
│   ├── SingleTextGraph.arxiv
│   │   ├── arxiv_CS_categories.txt
│   │   ├── labelidx2arxivcategeory.csv.gz
│   │   ├── nodeidx2paperid.csv.gz
│   │   ├── ogbn_arxiv
│   │   │   ├── mapping
│   │   │   │   ├── labelidx2arxivcategeory.csv.gz
│   │   │   │   ├── nodeidx2paperid.csv.gz
│   │   │   │   └── README.md
│   │   │   ├── processed
│   │   │   │   ├── geometric_data_processed.pt
│   │   │   │   ├── pre_filter.pt
│   │   │   │   └── pre_transform.pt
│   │   │   ├── raw
│   │   │   │   ├── edge.csv.gz
│   │   │   │   ├── node-feat.csv.gz
│   │   │   │   ├── node-label.csv.gz
│   │   │   │   ├── node_year.csv.gz
│   │   │   │   ├── num-edge-list.csv.gz
│   │   │   │   └── num-node-list.csv.gz
│   │   │   ├── RELEASE_v1.txt
│   │   │   └── split
│   │   │       └── time
│   │   │           ├── test.csv.gz
│   │   │           ├── train.csv.gz
│   │   │           └── valid.csv.gz
│   │   ├── processed
│   │   │   ├── data.pt
│   │   │   ├── geometric_data_processed.pt
│   │   │   ├── pca_64.pt
│   │   │   ├── pre_filter.pt
│   │   │   ├── pre_transform.pt
│   │   │   └── texts.pkl
│   │   └── raw
│   └── SingleTextGraph.WikiCS
│       ├── metadata.json
│       ├── processed
│       │   ├── data.pt
│       │   ├── data_undirected.pt
│       │   ├── geometric_data_processed.pt
│       │   ├── pca_64.pt
│       │   ├── pre_filter.pt
│       │   ├── pre_transform.pt
│       │   └── texts.pkl
│       └── raw
│           └── data.json
└── pyg
    └── Planetoid.PubMed
        ├── processed
        │   ├── data.pt
        │   ├── pca_64.pt
        │   ├── pre_filter.pt
        │   └── pre_transform.pt
        ├── PubMed
        │   ├── processed
        │   │   ├── data.pt
        │   │   ├── pre_filter.pt
        │   │   └── pre_transform.pt
        │   └── raw
        │       ├── ind.pubmed.allx
        │       ├── ind.pubmed.ally
        │       ├── ind.pubmed.graph
        │       ├── ind.pubmed.test.index
        │       ├── ind.pubmed.tx
        │       ├── ind.pubmed.ty
        │       ├── ind.pubmed.x
        │       └── ind.pubmed.y
        └── raw
```

# pretrain

``python pretrain.py``

# In-Context Learning on downstream graphs on node classification

``python icl_enhanced.py --dataset cora --k_shot 3 --n_runs 10 --gpu_id 0``


Sometimes raise ```RuntimeError: unable to open shared memory object </torch_3944815_4091667407_461> in read-write mode: Too many open files (24)``` 

Reducing the number of work in ```pt_model.py``` to 0 as:
```
def train_dataloader(self):
    return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, 
                        num_workers=0, pin_memory=True, collate_fn= lambda x: x)

def val_dataloader(self):
    return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn= lambda x: x)

def test_dataloader(self):
    return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True, collate_fn= lambda x: x)
```

if facing some converagence issue on some GPUs, just **reduce the learning rate from 0.01 to 0.005**, **enlarge the weight decay from 0.00005 to 0.0005**, and increase the pretraining epochs from 6000 to 8000~10000


## In-Context Node Classification
Cora: 
```python run_ICL_node.py --dataset cora --k_shot {k_shot}```  
k_shot = {1,3,5}

Computers:
```python run_ICL_node.py --dataset computers --k_shot {k_shot} --norm_feat```
k_shot = {1,3}
```python run_ICL_node.py --dataset computers --k_shot 5```