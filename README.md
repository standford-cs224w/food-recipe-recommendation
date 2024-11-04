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
In order to generate graph datasets, you can run the following python script:
```
python3 -m graph_generation.graph_dataset_generator
``` 

The graph generation contains the following steps:
1. Downloading data:
We download raw Food.com Recipes and Interactions files (`shuyangli94/food-com-recipes-and-user-interactions`) from Kaggle.

2. Processing users:
In order to generate user nodes, we only retrieve their IDs from `interactions_train.csv` as they don`t contain rich properties.
It should be noted `interactions_train.csv` contain all the users, hence there is no unseen users for validation and testing. We tested it out by running the following script:
    ```
    python3 -m graph_generation.compare_users
    ```

3. Processing recipes:
In order to generate recipes nodes, we use `RAW_recipes.csv` to process all the following selected properties with proper encoders:
    - `name`: `SequenceEncoder`
    - `steps`: `SequenceListEncoder`
    - `tags`: `SequenceListEncoder`
    - `ingredients`: `SequenceListEncoder`
    - `nutrition`: `ListEncoder`
    - `n_steps`: `IdentityEncoder`
    - `minutes`: `IdentityEncoder`

4. Processing edges:
In this step, we process edges with a single property (`rating`) for each data split separately. 
    - `interactions_train.csv` contains train edges
    - `interactions_validation.csv` contains validation edges   
    - `interactions_test.csv` contains test edges   

5. Generate graphs
In this step, we generate graph dataset for each data split given the processed users, recipes and their respective edges.

6. Storing graphs
Finally, we store the generated graphs in PyTorch (.pt) files which can be used to training, validation and testing later. 