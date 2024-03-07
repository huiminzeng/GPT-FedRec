# Federated Recommendation via Hybrid Retrieval Augmented Generation

This repository is the PyTorch impelementation for the paper "Federated Recommendation via Hybrid Retrieval Augmented Generation".

<img src=media/fig1.jpg width=400>

We propose GPT-FedRec, a federated recommendation framework leveraging ChatGPT and a novel hybrid Retrieval Augmented Generation (RAG) mechanism. GPT-FedRec is a two-stage solution. The first stage is a hybrid retrieval process, mining ID-based user patterns and text-based item features. Next, the retrieved results are converted into text prompts and fed into GPT for re-ranking. Our proposed hybrid retrieval mechanism and LLM-based re-rank aims to extract generalized features from data and exploit pretrained knowledge within LLM, overcoming data sparsity and heterogeneity in FR. In addition, the RAG approach also prevents LLM hallucination, improving the recommendation performance for real-world users.

<img src=media/fig2.jpg>

<!-- ## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{zeng2022attacking,
  title={On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks},
  author={Zeng, Huimin and Yue, Zhenrui and Zhang, Yang and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  year={2022},
  organization={IJCAI}
}
``` -->

## Requirements

For our running environment see requirements.txt

## Datasets
- For Beauty, Games, Auto and ML-100K, they will be automatically downloaded and processed, when the training scripts are executed.
- For Toys, please go to this [repo](https://github.com/jeykigung/P5?tab=readme-ov-file)
   - download the data, and place `datamaps.json`, `meta.json.gz`, `sequential_data.txt` into the `toys_new` folder
   - we call it `toys_new`, because it is already processed.
- Example folder structure
```
    ├── ...
    ├── data                   
    │   ├── auto 
    │   ├── beauty
    │   ├── toys_new
    │   └── ...
    ├── dataloader
    │   └── ...
    ├── datasets
    │   └── ...
    └── ...
```
## Scripts.

- Stage 1: train LRURec and E5 (Hybrid Retrieval)
   - Example
       ```
       python train_lru.py --dataset_code  --num_clients  --num_samples  --global_epochs  --local_epochs  --lr 
    
       python train_e5.py --dataset_code  --num_clients  --num_samples  --global_epochs --local_epochs  --lr
       ```
   - Hyperparameters
      ```
      --dataset_code            # select from 'beauty', 'games', 'toys_new', 'auto', 'ml-100k'
      --num_clients             # number of federated clients
      --num_samples             # number of training users per client
      --global_epochs           # global epochs for federated learning
      --local_epochs            # local epochs for federated learning
      --lr                      # learning rate
      ```
    - After executing the script, the test scores will be automatically saved to a new folder `experiments`

- Stage 2: Perform re-rank with GPT-3.5 (RAG-based Re-ranking)
    - Example
       ```
       python fuse.py --dataset_code  --openai_token --num_clients  --num_samples --lambda_ensemble 
       ```
   - Hyperparameters
      ```
      --openai_token            # the token to call the OpenAI API
      --dataset_code            # select from 'beauty', 'games', 'toys_new', 'auto', 'ml-100k'
      --num_clients             # number of federated clients
      --num_samples             # number of training users per client
      --lambda_ensemble         # the trade-off factor when computing the hybrid retrival score (Equation 4 in our paper)
      ```   
    - After executing the script, the generated texts will be saved to a new folder `llm_results`.

- Evaluation: 
    - Example
       ```
       python eval_fused.py --dataset_code  --num_clients  --num_samples  --lambda_ensemble 

       ```
    - After executing the script, the Recall and NDCG scores will be printed.


## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper.

<img src=media/results.jpg width=800>

## Acknowledgement

During the implementation, we base our code mostly on existing repos (e.g., LRURec). Many thanks to these authors for their great work!