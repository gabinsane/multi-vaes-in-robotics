# Multimodal VAEs in Robotics

This is the official code for the IROS 2024 submission "[Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks](https://arxiv.org/abs/2404.01932)".

We include implementations of the [MVAE](https://github.com/mhw32/multimodal-vae-public) 
([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) 
([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) 
([paper](https://openreview.net/forum?id=5Y21V0RDBV)) models.

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
```

## Dataset download 
You can download the datasets based on their codes. The codes are the following (explanations are provided in the paper):

![Coding 1](https://github.com/gabinsane/multi-vaes-in-robotics/blob/main/data2.png "dataset codes")

![Coding 2](https://github.com/gabinsane/multi-vaes-in-robotics/blob/main/data1.png "dataset codes")


The dataset should be placed in the ./data/lanro directory. For downloading, unzipping and moving the chosen dataset, run:

```
cd ~/multi-vaes-in-robotics/
wget https://data.ciirc.cvut.cz/public/groups/incognite/LANRO/D1a.zip   # replace d1a with lowercase codes from the tables above
unzip D1a.zip -d ./data/lanro 
```

## Setup and training

You can run the training with the chosen config as follows (assuming you downloaded or generated the dataset):

```
cd ~/multi-vaes-in-robotics/
python main.py --cfg configs/mmvae/d1a/config_lanro.yml
```

We provide configs for the experiments mentioned in the paper in the configs/ folder (sorted according to models and datasets). 


The config contains general arguments and modality-specific arguments (denoted as "modality_n"). In general, you can set up a training for 1-N modalities by defining the required subsections for each of them. 

The usage and possible options for all the config arguments are below (this is an example for another dataset called CdSprites+):

![Config documentation](https://data.ciirc.cvut.cz/public/groups/incognite/CdSprites/config2.png "config documentation")



## Evaluation

After training, you will find various visualizations of the training progress in the _./visuals_ folder of your experiment.
Furthermore, to evaluate on LANRO, you can choose one of the two scenarios:

```
cd ~/multi-vaes-in-robotics/models
python lanro_test.py --model modelpath --dataset 2  # specify the path to the model checkpoint and the dataset level (1-4) on which the model was trained
```
The code will run an evaluation on 500 trials and provide the successful_percentage.txt file in the model folder next to the .ckpt file.

## License

This code is published under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

## Citation

```
@inproceedings{sejnova2024bridginglanguagevisionaction,
      title={Bridging Language, Vision and Action: Multimodal VAEs in Robotic Manipulation Tasks}, 
      author={Gabriela Sejnova and Michal Vavrecka and Karla Stepanova},
      year={2024},
      booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      organization={IEEE}
}
```



## Acknowledgment

This code is adapted from the [Multimodal VAE Comparison toolkit](https://github.com/gabinsane/multimodal-vae-comparison).
The toolkit features models and functions from the official implementations of [MVAE](https://github.com/mhw32/multimodal-vae-public) ([paper](https://arxiv.org/abs/1802.05335)), [MMVAE](https://github.com/iffsid/mmvae) ([paper](https://arxiv.org/pdf/1911.03393.pdf)) and [MoPoE](https://github.com/thomassutter/MoPoE) ([paper](https://openreview.net/forum?id=5Y21V0RDBV)).
To generate the datasets and evaluate the models, we used an adapted version of the [LANRO simulator](https://github.com/frankroeder/lanro-gym).

## Contact

For any additional questions, feel free to email [sejnogab@fel.cvut.cz](mailto:sejnogab@fel.cvut.cz) 
