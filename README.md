# HeaRT

Official code for the paper ["Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking"](https://arxiv.org/pdf/2306.10453.pdf).


## Installation

The experiments were run using python 3.9. The required packages and their versions can be installed via the `requirements.txt` file. 
```
pip install -r requirements.txt
``` 

One exception is the original PEG code (`benchmarking/baseline_models/PEG`), which were run using python 3.7 and the packages in the `peg_requirements.txt` file.


## Download Data

All data can be downloaded via the following:
```
cd HeaRT  # Must be in the root directory
curl https://cse.msu.edu/~shomerha/HeaRT-Data/dataset.tar.gz | tar -xvz
``` 
This includes the negative samples generated by HeaRT and the splits for Cora, Citeseer, and Pubmed. The data for the OGB datasets will be automatically downloaded from the `ogb` package.

Please note that the resulting directory `dataset` must be placed in the root project directory.



## Reproduce Results

To reproduce the results, please refer to the settings in the **scripts/hyparameters** directory. We include a file for each dataset which includes the command to train and evaluate each possible method.

For example, to reproduce the results on ogbl-collab under the existing evaluation setting, the command for each method can be found in the `ogbl-collab.sh` file located in the `scripts/hyperparameter/existing_setting_ogb/` directory. 

To run the code, we need to first go to the setting directory:
- benchmarking/exist_setting_small: running files for cora, citeseer, and pubmed under the existing setting
- benchmarking/exist_setting_ogb: running files for ogbl-collab, ogbl-ppa, and ogbl-citation2 under the existing setting
- benchmarking/exist_setting_ddi: running files for ogbl-ddi under the existing setting
- benchmarking/HeaRT_small: running files for cora, citeseer, and pubmed under HeaRT
- benchmarking/HeaRT_ogb: running files for  ogbl-collab, ogbl-ppa, and ogbl-citation2 under HeaRT
- benchmarking/HeaRT_ddi/: running files for ogbl-ddi under HeaRT

One example to run GCN  on cora under the **existing setting** is shown below (similar for citeseer and pubmed) :
```
cd benchmarking/exist_setting_small/
python  main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1  --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024
```
**
One example to run GCN  on ogbl-collab under the **existing setting** is shown below (similar for ogbl-ppa and ogbl-citation2):
```
cd benchmarking/exist_setting_ogb/
python main_gnn_ogb.py  --use_valedges_as_input  --data_name ogbl-collab  --gnn_model GCN --hidden_channels 256 --lr 0.001 --dropout 0.  --num_layers 3 --num_layers_predictor 3 --epochs 9999 --kill_cnt 100  --batch_size 65536 
```

One example to run GCN on ogbl-ddi under the **existing setting** is shown below:
```
cd benchmarking/exist_setting_ddi/
python main_gnn_ddi.py --data_name ogbl-ddi --gnn_model GCN  --lr 0.01 --dropout 0.5  --num_layers 3 --num_layers_predictor 3  --hidden_channels 256 --epochs 9999 --eval_steps 1 --kill_cnt 100 --batch_size 65536 
```

<!-- One example to run GCN  on cora under **HeaRT** is shown below (similar for citeseer and pubmed) :
```
cd benchmarking/HeaRT_small/
python main_gnn_CoraCiteseerPubmed.py  --data_name cora  --gnn_model GCN  --lr 0.001 --dropout 0.5 --l2 0 --num_layers 1 --hidden_channels 256  --num_layers_predictor 3  --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024 
``` -->


## Generate Negative Samples using HeaRT

The set of negative samples generated by HeaRT, that were used in the study, can be reproduced via the scripts in the `scripts/HeaRT/` directory. 

A custom set of negative samples can be produced by running the `heart_negatives/create_heart_negatives.py` script. Multiple options exist to customize the negative samples. This includes:
- The CN metric used. Can be either `CN` or `RA` (default is `RA`). Specified via the `--cn-metric` argument.
- The aggregation function used. Can be either `min` or `mean` (default is `min`). Specified via the `--agg` argument.
- The number of negatives generated per positive sample. Specified via the `--num-samples` argument (default is 500).
- The PPR parameters. This includes the tolerance used for approximating the PPR (`--eps` argument) and the teleporation probability (`--alpha` argument). `alpha` is fixed at 0.15 for all datasets. For the tolerance, `eps`, we recommend following the settings found in `scripts/HeaRT`.


## Cite

