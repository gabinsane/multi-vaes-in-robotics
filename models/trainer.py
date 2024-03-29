from torch import optim, nn
import torch
import pytorch_lightning as pl
import models
from models.config_cls import Config
from models.vae import VAE
import os, copy
from utils import make_kl_df, data_to_device, check_input_unpacked, make_joint_samples
from models.mmvae_base import TorchMMVAE
from visualization import plot_kls_df
from visualization import t_sne


class MultimodalVAE(pl.LightningModule):
    """
    Multimodal VAE trainer common for all architectures. Configures, trains and tests the model.

    :param feature_dims: dictionary with feature dimensions of training data
    :type feature_dims: dict
    :param cfg: instance of Config class
    :type cfg: object
    """

    def __init__(self, cfg, feature_dims: dict):
        super().__init__()
        self.config = self.check_config(cfg)
        self.optimizer = None
        self.objective = None
        self.datamodule = None
        self.mod_names = self.get_mod_names()
        self.feature_dims = feature_dims
        self.get_model()
        self.example_input_array = None
        self.latents = cfg.n_latents

    def check_config(self, cfg):
        """
        Creates a Config class out of the provided argument parser

        :param cfg: argument parser or str path to config
        :type cfg: (argparse.ArgumentParser, str)
        :return: Config instance
        :rtype: object
        """
        if not isinstance(cfg, models.config_cls.Config):
            cfg = Config(cfg)
        return cfg

    def get_mod_names(self):
        """
        Creates a dictionary with modality numbers and their names based on dataset

        :return: Dict with modality numbers as keys and names as values
        :rtype: dict
        """
        mod_names = {}
        for i, m in enumerate(self.config.mods):
            mod_names["mod_{}".format(i + 1)] = m["mod_type"]
        return mod_names

    @property
    def datamod(self):
        """
        When the class is used for inference, there is no pl trainer module

        :return: an instance of DataModule class
        :rtype:
        """
        try:
            return self.trainer.datamodule
        except:
            return self.datamodule

    def configure_optimizers(self):
        """
        Sets up the optimizer specified in the config
        """
        if self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=float(self.config.lr), amsgrad=True)
        elif self.config.optimizer.lower() == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(self.model.parameters(), lr=float(self.config.lr), eps=1e-16,
                                       betas=(0.9, 0.999),
                                       weight_decouple=True, rectify=False, print_change_log=False)
        else:
            raise NotImplementedError
        return self.optimizer

    def get_model(self):
        """
        Sets up the model according to the config file
        """
        if self.config.pre_trained:
            self.model = self.load_from_checkpoint(self.config.pre_trained, cfg=self.config)
            return self.model
        vaes = {}
        for i, m in enumerate(self.config.mods):
            enc_mu_logvar = False if self.config.mixing == "htvae" else True
            vaes["mod_{}".format(i + 1)] = VAE(m["encoder"], m["decoder"], self.feature_dims[m["mod_type"]],
                                               self.config.n_latents, m["recon_loss"], m["private_latents"],
                                               obj_fn=self.config.obj,
                                               beta=self.config.beta, id_name="mod_{}".format(i + 1),
                                               prior_dist=m["prior"], post_dist=m["prior"], likelihood_dist=m["prior"],
                                               llik_scaling=m["llik_scaling"], enc_mu_logvar=enc_mu_logvar)
        if len(self.config.mods) > 1:
            vaes = nn.ModuleDict(vaes)
            obj_cfg = {"obj": self.config.obj, "beta": self.config.beta, "K":self.config.K}
            self.model = getattr(models, self.config.mixing.lower())(vaes, self.config.n_latents, obj_cfg,
                                                                     self.config.model_cfg)
            assert isinstance(self.model, TorchMMVAE)
        else:
            self.model = vaes["mod_1"]  # unimodal VAE scenario
        self.save_hyperparameters()
        return self.model

    def get_forbidden_mod_combos(self):
        forbidden_combos = self.datamod.datasets[0].forbidden_subsets
        for i, s in enumerate(forbidden_combos):
            for key, value in self.mod_names.items():
                if value in forbidden_combos[i]:
                    forbidden_combos[i] = s.replace(value, key)
        return forbidden_combos

    def training_step(self, train_batch, batch_idx):
        """
        Iterates over the train loader
        """
        loss_d = self.model.objective(train_batch, self.get_forbidden_mod_combos())
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("train_{}".format(key), loss_d[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss_d[key]):
                    self.log("Mod_{}_TrainLoss".format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_step(self, val_batch, batch_idx):
        """
        Iterates over the val loader
        """
        loss_d = self.model.objective(val_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("val_{}".format(key), loss_d[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss_d[key]):
                    self.log("Mod_{}_ValLoss".format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def test_step(self, test_batch, batch_idx):
        """
        Iterates over the test loader (if test data is provided, otherwise val loader)
        """
        loss_d = self.model.objective(test_batch)
        for key in loss_d.keys():
            if key != "reconstruction_loss":
                self.log("test_{}".format(key), loss_d[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss_d[key]):
                    self.log("Mod_{}_TestLoss".format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss_d["loss"]

    def validation_epoch_end(self, outputs):
        """
        Save visualizations at the end of validation epoch

        :param outputs: Loss that comes from validation_step
        :type outputs: torch.tensor
        """
        if (self.trainer.current_epoch) % self.config.viz_freq == 0:
            savepath = os.path.join(self.config.mPath, "visuals/epoch_{}/".format(self.trainer.current_epoch))
            os.makedirs(savepath, exist_ok=True)
            self.analyse_data(savedir=savepath)
            self.save_reconstructions(savedir=savepath)
            try:
                self.save_joint_samples(savedir=savepath, num_samples=10, traversals=True)
                self.save_joint_samples(num_samples=64, savedir=savepath, traversals=False)
            except:
                pass


    def test_epoch_end(self, outputs):
        """Visualizations to make at the end of the testing epoch"""
        savepath = os.path.join(self.config.mPath, "visuals/epoch_{}_test/".format(self.trainer.current_epoch))
        os.makedirs(savepath, exist_ok=True)
        self.analyse_data(savedir=savepath, split="test")
        self.save_reconstructions(savedir=savepath, split="test")
        if self.datamod.datasets[0].eval_statistics_fn() is not None:
            self.datamod.datasets[0].eval_statistics_fn()(self)

    def save_reconstructions(self, num_samples=10, savedir=None, split="val"):
        """
        Reconstructs data and saves output, also iterates over missing modalities on the input to cross-generate

        :param num_samples: number of samples to take from the dataloader for reconstruction
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param split: val/test, whether to take samples from test or validation dataloader
        :type split: str
        """

        def save(output, mods, name=None):
            for k in data.keys():
                if mods[k]["data"] is None:
                    mods.pop(k)
            for key in output.mods.keys():
                recon_list = [x.loc for x in output.mods[key].decoder_dist] if isinstance(output.mods[key].decoder_dist,
                                                                                          list) \
                    else output.mods[key].decoder_dist.loc
                data_class = self.datamod.datasets[int(key.split("_")[-1]) - 1]
                if data_class.mod_type == name.split("_")[-1] or len(self.config.mods) == 1:
                    p = os.path.join(savedir, "recon_{}_to_{}.png".format(name, data_class.mod_type))
                    data_class.save_recons(mods, recon_list, p, self.mod_names)

        data, labels = self.datamod.get_num_samples(num_samples, split=split)
        data_i = check_input_unpacked(data_to_device(data, self.device))
        save(self.model.forward(data_i), data, "all")
        if len(self.config.mods) > 1:
            output = self.model.forward(check_input_unpacked(data))
            save(output, copy.deepcopy(data), "language")
            save(output, copy.deepcopy(data), "shapes")
            save(output, copy.deepcopy(data), "colors")
            for m in range(len(data.keys())):
                mods = copy.deepcopy(data)
                mods["mod_{}".format(m + 1)]["data"], mods["mod_{}".format(m + 1)]["masks"] = None, None
                # for d in mods.keys():
                #     mods[d]["data"], mods[d]["masks"] = None, None
                #mods["mod_{}".format(m + 1)] = data["mod_{}".format(m + 1)]
                mod_name = "all_but_{}".format(self.config.mods[m]["mod_type"])
                output = self.model.forward(check_input_unpacked(mods))
                save(output, copy.deepcopy(mods), mod_name)

    def save_joint_samples(self, num_samples=16, savedir=None, traversals=False):
        """
        Generate joint samples from random vectors and save them

        :param num_samples: number of samples to generate
        :type num_samples: int
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param traversals: whether to make traversals for each dimension (True) or randomly sample latents (False)
        :type traversals: bool
        """
        recons = {}
        traversal_ranges = [(-6,6), (-4,4), (-2,2), (-1,1)]
        for rng in traversal_ranges:
            if len(self.config.mods) > 1:
                for i, vae in enumerate(self.model.vaes):
                    n_latents = self.model.vaes[vae].total_latents
                    recons["mod_{}".format(i+1)], recons["mod_{}_raw".format(i+1)] = make_joint_samples(self.model, i, self.datamod, n_latents, traversals,
                                                                          savedir, num_samples, trav_range=rng, current_vae=vae)
            else:
                recons["mod_1"], recons["mod_1_raw"] = make_joint_samples(self.model, 0, self.datamod, self.model.total_latents, traversals,
                                                     savedir, num_samples, trav_range=rng)
        return recons


    def analyse_data(self, data=None, labels=None, num_samples=250, path_name="", savedir=None, split="val", fn_list=["tsne"], device=None):
        """
        Encodes data and plots T-SNE. If no data is passed, a dataloader (based on split="val"/"test") will be used.

        :param data: test data
        :type data: torch.tensor
        :param labels: labels for the data for labelled T-SNE (optional) - list of strings
        :type labels: list
        :param num_samples: number of samples to use for visualization
        :type num_samples: int
        :param path_name: label under which to save the visualizations
        :type path_name: str
        :param savedir: where to save the reconstructions
        :type savedir: str
        :param split: val/test, whether to take samples from test or validation dataloader
        :type split: str
        """
        if not data:
            data, labels = self.datamod.get_num_samples(num_samples, split=split)
        output_dic = self.eval_forward(data, device=device)
        pz = self.model.get_pz([x for x in self.model.pz_params])
        zss_sampled = [pz.sample(torch.Size([1, num_samples])).view(-1, pz.batch_shape[-1]),
                       *[zs["latents"].view(-1, zs["latents"].size(-1)) for zs in output_dic["latent_samples"]]]
        if path_name == "" and not self.config.eval_only:
            path_name = "_e_{}".format(self.trainer.current_epoch)
        if "kld" in fn_list:
             kl_df = make_kl_df(output_dic["encoder_dist"], pz)
             plot_kls_df(kl_df, os.path.join(savedir, 'kl_distance{}.png'.format(path_name)))
        if "tsne" in fn_list:
            inp = [x for x in zss_sampled[1:]]
            # if self.config.obj == "elbo_gumbel":
            #      for id, i in enumerate(inp):
            #          inp[id] = torch.argmax(i.reshape(i.shape[0], -1, self.model.data_dim[1]), dim=-1)
            t_sne([x for x in zss_sampled[1:]], os.path.join(savedir, 't_sne{}.png'.format(path_name)), labels,
                  self.mod_names)
        return None

    def eval_forward(self, data, device=None):
        device = device if device is not None else self.device
        data_i = check_input_unpacked(data_to_device(data, device))
        output = self.model.forward(data_i)
        output_dic = output.unpack_values()
        return output_dic
