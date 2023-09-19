# Multimodal VAEs in Robotics

This is the official code for the ICRA 2024 submission "Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks".

We provide implementations of the [MVAE](https://github.com/mhw32/multimodal-vae-public) 
([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) 
([paper](https://arxiv.org/pdf/1911.03393.pdf)), [MoPoE](https://github.com/thomassutter/MoPoE) 
([paper](https://openreview.net/forum?id=5Y21V0RDBV)) and [DMVAE](https://github.com/seqam-lab/DMVAE) ([paper](https://github.com/seqam-lab/DMVAE)) models.

## Preliminaries

This code was tested with:

- Python version 3.8.13
- PyTorch version 1.12.1
- CUDA version 10.2 and 11.6

We recommend to install the conda enviroment as follows:

```
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
conda activate multivae                 
```

For evaluation, you will also need to install our local, adapted version of the [LANRO simulator](https://github.com/frankroeder/lanro-gym):

```
pip install gymnasium
cd models/lanro_gym
pip install -e .
```

## Dataset download 
You can download any of the following difficulty levels: D1, D2, D3 and D4
The dataset should be placed in the ./data/lanro directory. For downloading, unzipping and moving the chosen dataset, run:

```
cd ~/multi-vaes-in-robotics/
wget https://data.ciirc.cvut.cz/public/groups/incognite/LANRO/D1.zip   # replace D1 with D2, D3 or D4 for each dataset
unzip D1.zip -d ./data/lanro   # replace D1 with D2, D3 or D4 for each dataset
```

## Setup and training

You can run the training with the chosedn config as follows (assuming you downloaded or generated the dataset):

```
cd ~/multi-vaes-in-robotics/
python main.py --cfg configs/mmvae/config_lanro_d1.yml
```

We provide configs for the experiments mentioned in the paper in the configs/ folder (sorted according to models and datasets). 


The config contains general arguments and modality-specific arguments (denoted as "modality_n"). In general, you can set up a training for 1-N modalities by defining the required subsections for each of them. 

The usage and possible options for all the config arguments are below:

![Config documentation](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/config2.png "config documentation")



## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to evaluate on LANRO, you can choose one of the two scenarios:

### Generating Actions from Images and NL commands

```
cd ~/multi-vaes-in-robotics/
python models/lanro_test.py --model modelpath --dataset 2  # specify the path to the model checkpoint and the dataset level (1-4) on which the model was trained
```

The code will run evaluation on the testset of the dataset. Those are 1200 trials. If you want to use only e.g. every 100th sample, you can add the `--subsample 100` argument

### Generating NL commands from Images and Actions

```
cd ~/multi-vaes-in-robotics/
python models/lanro_test_language.py --model modelpath --dataset 2  # specify the path to the model checkpoint and the dataset level (1-4) on which the model was trained
```

The code will run evaluation on the testset of the dataset. Those are 1200 trials. If you want to use only e.g. every 100th sample, you can add the `--subsample 100` argument


## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  


## Acknowledgment

This code is adapted from the [Multimodal VAE Comparison toolkit](https://github.com/gabinsane/multimodal-vae-comparison).
The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).
To generate the datasets and evaluate the models, we used an adapted version of the [LANRO simulator](https://github.com/frankroeder/lanro-gym).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
