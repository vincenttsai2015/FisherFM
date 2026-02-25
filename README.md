# Fisher Flow Matching
All our dependencies are listed in `environment.yaml`, for Conda, and `requirements.txt`, for `pip`. Please also separately install `DGL`:
```bash
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```
Our code contains parts of [FlowMol](https://github.com/Dunni3/FlowMol/tree/main) by Dunn and Koes [1] (most of QM9 experiments), [Riemannian-FM](https://github.com/facebookresearch/riemannian-fm) by Chen, et al. [2], and, for the baselines, [DFM](https://github.com/HannesStark/dirichlet-flow-matching/tree/main) by Stark, et al [3].

## Installation
* Create the environment: ```conda create -c conda-forge -n FisherFM rdkit=2023.9.5 python=3.10```
* Activate the environment: ```conda activate FisherFM```
* Verify the installation of rdkit: ```python -c 'from rdkit import Chem'```
* Install Hydra: ```python -m pip install hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0```
* Install selene-sdk (0.6.0): ```python -m pip install selene-sdk```
    * There occured an error when installing version 0.4.4
* Install PyTorch: ```python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121```
* Install PyG: 
    * ```python -m pip install torch_geometric```
    * ```python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html```
* Install PyTorch-lightning and torchmetrics: ```python -m pip install lightning torchmetrics```
* Install torchdiffeq and torch-ema: ```python -m pip install torchdiffeq torch-ema```
* Install DGL: ```python -m pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu121/repo.html```
* Install setuptools with version<81: ```python -m pip install "setuptools<81"```
    * Verify setuptools ```python -c "import pkg_resources; print('OK')"```
* Install other required packages: ```python -m pip install pandas POT dirichlet einops biopython pyBigWig pyranges cooler cooltools wandb rootutils pre-commit rich pytest geoopt einops geomstats ipdb transformers schedulefree pydantic diffusers```

## Toy Experiment
For the DFM toy experiment, the following command allows us to run our code:
```bash
python -m src.train experiment=toy_dfm_bmlp data.dim=100 trainer=gpu trainer.max_epochs=500
```
Of course, the dimension argument is varied, and the configuration files allow for changing manifolds (`"simplex"`, or `"sphere"`) and turn OT on/off (`"exact"` or `"None"`).

## Promoter and Enhancer DNA Experiment
To download the datasets, it suffices to follow the steps of [Stark, et al](https://github.com/HannesStark/dirichlet-flow-matching/). For evaluating the FBD, it also needed to download their weights from their `workdir.zip`. To run the promoter dataset experiment, the following command can be used:

```bash
python -m src.train experiment=promoter_sfm_promdfm trainer.max_epochs=200 trainer=gpu data.batch_size=128
```

As for the enhancer MEL2 experiment, the following command is available:

```bash
python -m src.train experiment=enhancer_mel_sfm_cnn trainer.max_epochs=800 trainer=gpu
```

and for the FlyBrain DNA one:
```bash
python -m src.train experiment=enhancer_fly_sfm_cnn trainer.max_epochs=800 trainer=gpu
```

## QM9 experiment
To install the QM9 dataset, we have included `process_qm9.py` from FlowMol, so it suffices to follow the steps indicated in their [README](https://github.com/Dunni3/FlowMol/tree/main).

```bash
python -m src.train experiment=qm_clean_sfm trainer=gpu
```

## References
- [1]: [Dunn and Koes: Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation](https://arxiv.org/abs/2404.19739).
- [2]: [Chen, et al.: Flow Matching on General Geometries](https://arxiv.org/pdf/2302.03660).
- [3]: [Stark, et al.: Dirichlet Flow Matching with Applications to DNA Sequence Design](https://arxiv.org/abs/2402.05841).
