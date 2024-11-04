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
    We download raw Food.com Recipes and Interactions files (`shuyangli94/food-com-recipes-and-user-interactions`) via KaggleAPI.  
    Please note `KAGGLE_USERNAME` and `KAGGLE_KEY` needs to be set in the .env file.

2. Processing users:  
    In order to generate user nodes, we only retrieve their IDs from `interactions_train.csv` as they do not contain rich properties.  
    Notes:
    - Originally, we intented to consider `techniques` as a property for user ndoes, but they seem to be based on all the edges (interactions). Considering we split the edges into training, validation, and test sets, adding `techniques` might lead to biased evaluation results.

    - `interactions_train.csv` contains all the users, hence there is no unseen users for validation and testing. We tested it out by running the following script:
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

    In generall, the encoders make sure the values are in numeric form with correect data types and also convert them into Torch tensors to prepare them for training and inference.

4. Processing edges:  
    In this step, we process edges with a single property (`rating`) for each data split separately. 
    - `interactions_train.csv` contains train edges
    - `interactions_validation.csv` contains validation edges   
    - `interactions_test.csv` contains test edges   

5. Generating graphs:  
    In this step, we generate graph dataset for each data split given the processed users, recipes and their respective edges.

6. Storing graphs:  
    Finally, we store the generated graphs in PyTorch (.pt) files which can be used for training, validation and testing later. 