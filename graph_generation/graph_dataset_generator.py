import os

from ast import literal_eval
from typing import Optional, Any

import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv; load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit


class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to PyTorch tensors.
    def __init__(self, dtype: Optional[torch.dtype] = None) -> None:
        self.dtype = dtype

    def __call__(self, values: pd.Series) -> torch.Tensor:
        return torch.from_numpy(values.to_numpy()).view(-1, 1).to(self.dtype)


class ListEncoder:
    # The 'SequenceEncoder' encodes a raw column of list of a given type into tensors.
    def __init__(self, dtype: Optional[torch.dtype] = None) -> None:
        self.dtype = dtype

    def __call__(self, values: pd.Series) -> torch.Tensor:
        return  torch.tensor(values.tolist(), dtype=self.dtype)    


class SequenceEncoder:
    # The 'SequenceEncoder' encodes raw column strings into embeddings tensors.
    def __init__(self, model: SentenceTransformer, batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, values: pd.Series) -> torch.Tensor:
        x = self.model.encode(
            values.to_numpy(), 
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=self.batch_size
        )
        return x.cpu()


class SequenceListEncoder:
    # The 'SequenceEncoder' encodes a raw column of list of strings into embeddings tensors.
    def __init__(self, model: SentenceTransformer, batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, values: pd.Series) -> torch.Tensor:
        x = self.model.encode(
            values.apply(', '.join).to_numpy(), 
            show_progress_bar=True,
            convert_to_tensor=True, 
            batch_size=self.batch_size
        )
        return x.cpu()


def download_raw_data(dataset_name: str, destination_path: str) -> None:
    """Download raw dataset files from Kaggle."""
    api = KaggleApi()
    # api.authenticate()

    api.dataset_download_files(dataset_name, path=destination_path, unzip=True)


def load_sequence_embedder(model_id: str, device: Optional[str] = None) -> SentenceTransformer:
    """Load a pre-trained embeddings model with the given ID from HugginFace for encoding sequences."""
    return SentenceTransformer(model_id, device=device)


def load_nodes_from_csv(file_path: str, index_col: str, encoders: Optional[dict[str, Any]] = None, **kwargs) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    df = pd.read_csv(file_path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = []
        for col, encoder in encoders.items():
            print(f'-Encoding {col}...')
            xs.append(encoder(df[col]))

        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edges_from_csv(file_path: str, src_index_col: str, src_mapping: list[tuple[int, int]], dst_index_col: str, dst_mapping: list[tuple[int, int]], encoders: Optional[dict[str, Any]] = None, **kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    df = pd.read_csv(file_path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_label = None
    if encoders is not None:
        edge_attrs = []
        for col, encoder in encoders.items():
            print(f'-Encoding {col}...')
            edge_attrs.append(encoder(df[col]))
    
        edge_label = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_label


def generate_graph(user_mapping: list[tuple[int, int]], recipe_x: torch.Tensor, edge_index: torch.Tensor, edge_label: Optional[torch.Tensor]) -> HeteroData:
    graph_data = HeteroData()
    graph_data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    graph_data['recipe'].x = recipe_x
    graph_data['user', 'rates', 'recipe'].edge_index = edge_index
    graph_data['user', 'rates', 'recipe'].edge_label = edge_label

    # Add reverse ('recipe', 'rev_rates', 'user') relation for message passing.
    graph_data = ToUndirected()(graph_data)
    # Remove 'reverse' label.
    del graph_data['recipe', 'rev_rates', 'user'].edge_label  

    return graph_data

def split_graph(graph_data: HeteroData, training_supervision_ratio: float, validation_ratio: float, test_ratio: float) -> dict[str, HeteroData]:
    transform = RandomLinkSplit(
        disjoint_train_ratio=training_supervision_ratio,
        num_val=validation_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'recipe')],
        rev_edge_types=[('recipe', 'rev_rates', 'user')],
    )

    train_graph, validation_graph, test_data = transform(graph_data)

    return {
        'train': train_graph, 
        'validation': validation_graph, 
        'test': test_graph
    }


def store_graph(graph: HeteroData, file_path: str) -> None:
    torch.save(graph, file_path)


if __name__ == '__main__':
    base_data_path = 'data'
    raw_dataset_name = 'shuyangli94/food-com-recipes-and-user-interactions'
    sequence_embedder_id = 'sentence-transformers/all-mpnet-base-v2'
    device = 'cuda' if torch.cuda.is_available() else None
    batch_size = 64
    training_supervision_ratio = 0.2
    validation_ratio = 0.05
    test_ratio = 0.1
    graph_version = 1

    print("Downloading raw files from Kaggle...")
    # download_raw_data(raw_dataset_name, base_data_path)

    print('Processing user nodes...')
    _, user_mapping = load_nodes_from_csv(
        f'{base_data_path}/RAW_interactions.csv', 
        index_col='user_id'
    )

    print('Processing recipe nodes...')
    sequence_embedder = load_sequence_embedder(sequence_embedder_id, device)
    recipe_x, recipe_mapping = load_nodes_from_csv(
        f'{base_data_path}/RAW_recipes.csv',
        index_col='id',
        encoders={
            'name': SequenceEncoder(model=sequence_embedder, batch_size=batch_size),
            'steps': SequenceListEncoder(model=sequence_embedder, batch_size=batch_size),
            'tags': SequenceListEncoder(model=sequence_embedder, batch_size=batch_size),
            'ingredients': SequenceListEncoder(model=sequence_embedder, batch_size=batch_size),
            'nutrition': ListEncoder(dtype=torch.float),
            'n_steps': IdentityEncoder(dtype=torch.long),
            'minutes': IdentityEncoder(dtype=torch.long)
        },
        converters={'steps': literal_eval, 'tags': literal_eval, 'ingredients': literal_eval, 'nutrition': literal_eval}
    )

    print('Processing edges...')
    rating_path = f'{base_data_path}/RAW_interactions.csv'
    edge_index, edge_label = load_edges_from_csv(
        file_path=rating_path,
        src_index_col='user_id',
        src_mapping=user_mapping,
        dst_index_col='recipe_id',
        dst_mapping=recipe_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )

    print(f'Generating graph dataset...')
    graph = generate_graph(
        user_mapping=user_mapping,
        recipe_x=recipe_x,
        edge_index=edge_index,
        edge_label=edge_label

    )

    print(f'Splitting graph dataset...')
    graph_splits = split_graph(graph, training_supervision_ratio, validation_ratio, test_ratio)

    print('Storing graph splits...')
    for split_name, graph in graph_splits.items():
        graph_path = f'{base_data_path}/graph/v{graph_version}/{split_name}_graph.pt'
        store_graph(graph, graph_path)
        print(f'{split_name} graph stored in {graph_path}.')
