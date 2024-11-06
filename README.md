# Comprehensive Analysis of GNN Approaches for Food Recipe Recommendation

## Environment Setup
1) Setup python virtual environement with required depedencies:
    ```
    sudo chmod +x install_requirements.sh

    bash install_requirements.sh

    source venv/bin/activate
    ```

2) Set environment variables
    ```
    cp .env.example .env
    ```

## Graph Dataset Generation
In order to generate graph datasets, you can run the following command:
```
python3 -m graph_generation.graph_dataset_generator
``` 

The details can be found in the [graph_dataset_generator.py](./graph_generation/graph_dataset_generator.py)


The graph generation contains the following steps:  
1. Downloading data:  
    We download raw Food.com Recipes and Interactions files (`shuyangli94/food-com-recipes-and-user-interactions`) via KaggleAPI.  
    Please note `KAGGLE_USERNAME` and `KAGGLE_KEY` needs to be set in the .env file.

2. Processing users:  
    In order to generate user nodes, we only retrieve their IDs from `RAW_interactions.csv` as they do not contain rich properties.  
    Notes:
    - Originally, we intented to consider `techniques` as a property for user ndoes, but they seem to be based on all the edges (interactions). Considering we split the edges into training, validation, and test sets, adding `techniques` might lead to biased evaluation results.

    - `RAW_interactions.csv` contains all the users, hence there are no unseen users for validation and testing.

3. Processing recipes:   
    In order to generate recipes nodes, we use `RAW_recipes.csv` to process all the following selected properties with proper encoders:
    - `name`: `SequenceEncoder`
    - `steps`: `SequenceListEncoder`
    - `tags`: `SequenceListEncoder`
    - `ingredients`: `SequenceListEncoder`
    - `nutrition`: `ListEncoder`
    - `n_steps`: `IdentityEncoder`
    - `minutes`: `IdentityEncoder`  

    In generall, the encoders make sure the values are in numeric form with correect data types and also convert them into Torch tensors to prepare them for training and inference.

4. Processing edges:  
    In this step, we process edges with a single property (`rating`) using `RAW_interactions.csv` dataset.

5. Generating graph:  
    In this step, we generate a heterogeneous graph dataset given the processed users, recipes and their respective edges.
    We construct heterogeneous graphs using `torch_geometric.data.HeteroData` since we have two different types of nodes (users and edges).

6. Generating data splits:
    After generating the graph, we split them into three graph with distinct edges using `torch_geometric.transforms.RandomLinkSplit`:
    - Train graph: It contains 85% of all edges, 20% percent of which is used as supervision edges.
    - Validation graph: It contains 5% of the remaining edges.
    - Test graph: It contains 10% of the remaining edges.  

    These are the hyperparameters in the code that can control the edge ratios for each split:
    ```
    supervision_train_ratio = 0.2
    validation_ratio = 0.05
    test_ratio = 0.1
    ```

6. Storing graphs:  
    Finally, we store the generated graphs in PyTorch (.pt) files which can be used for training, validation and testing later. 
