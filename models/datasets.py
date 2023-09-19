import torch
import cv2, os
import numpy as np
import math
import pickle
from utils import one_hot_encode, lengths_to_mask, turn_text2image, load_data, add_recon_title, one_hot_encode_words, seq2words
from torchvision.utils import make_grid
from itertools import compress


class BaseDataset():
    """
    Abstract dataset class shared for all datasets
    """

    def __init__(self, pth, testpth, mod_type):
        """

        :param pth: path to the given modality
        :type pth: str
        :param mod_type: tag for the modality for correct processing (e.g. "text", "image", "mnist", "svhn" etc.)
        :type mod_type: str
        """
        assert hasattr(self, "feature_dims"), "Dataset class must have the feature_dims attribute"
        self.path = pth
        self.testdata = testpth
        self.current_path = None
        self.mod_type = mod_type
        self.has_masks = False
        self.categorical = False

    def _mod_specific_loaders(self):
        """
        Assigns the preprocessing function based on the mod_type
        """
        raise NotImplementedError

    def _mod_specific_savers(self):
        """
        Assigns the postprocessing function based on the mod_type
        """
        raise NotImplementedError

    def labels(self):
        """Returns labels for the whole dataset"""
        return None

    def get_labels(self, split="train"):
        """Returns labels for the given split: train or test"""
        self.current_path = self.path if split == "train" else self.testdata
        return self.labels()

    def eval_statistics_fn(self):
        """(optional) Returns a dataset-specific function that runs systematic evaluation"""
        return None

    def current_datatype(self):
        """Returns whther the current path to data points to test data or train data"""
        if self.current_path == self.testdata:
            return "test"
        elif self.current_path == self.path:
            return "train"
        else:
            return None

    def _preprocess(self):
        """
        Preprocesses the loaded data according to modality type

        :return: preprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_loaders().keys(), "Unsupported modality type for {}".format(
            self.current_path)
        return self._mod_specific_loaders()[self.mod_type]()

    def _postprocess(self, output_data):
        """
        Postprocesses the output data according to modality type

        :return: postprocessed data
        :rtype: list
        """
        assert self.mod_type in self._mod_specific_savers().keys(), "Unsupported modality type for {}".format(self.current_path)
        return self._mod_specific_savers()[self.mod_type](output_data)

    def get_processed_recons(self, recons_raw):
        """
        Returns the postprocessed data that came from the decoders

        :param recons_raw: tensor with output reconstructions
        :type recons_raw: torch.tensor
        :return: postprocessed data as returned by the specific _postprocess function
        :rtype: list
        """
        return self._postprocess(recons_raw)

    def get_data_raw(self):
        """
        Loads raw data from path

        :return: loaded raw data
        :rtype: list
        """
        data = load_data(self.current_path)
        return data

    def get_data(self):
        """
        Returns processed data

        :return: processed data
        :rtype: list
        """
        self.current_path = self.path
        return self._preprocess()

    def get_test_data(self):
        """
        Returns processed test data if available

        :return: processed data
        :rtype: list
        """
        if self.testdata is not None:
            self.current_path = self.testdata
            return self._preprocess()
        return None

    def _preprocess_images(self, dimensions):
        """
        General function for loading images and preparing them as torch tensors

        :param dimensions: feature_dim for the image modality
        :type dimensions: list
        :return: preprocessed data
        :rtype: torch.tensor
        """
        data = [torch.from_numpy(np.asarray(x.reshape(*dimensions)).astype(np.float)) for x in self.get_data_raw()]
        return torch.stack(data)

    def _preprocess_text_onehot(self):
        """
        General function for loading text strings and preparing them as torch one-hot encodings

        :return: torch with text encodings and masks
        :rtype: torch.tensor
        """
        self.has_masks = True
        self.categorical = True
        data = []
        for x in self.get_data_raw():
            d = " ".join(x) if isinstance(x, list) else x
            data.append(d)
        self.lang_labels = data
        data = [one_hot_encode(len(f), f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type not in ["text", "atts"] else output_processed
        return output_processed

    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 3:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=num_dims))
            cv2.imwrite(path, cv2.cvtColor(np.transpose(grid, (1,2,0)).astype("uint8"), cv2.COLOR_BGR2RGB))
        else:
            output_processed = torch.stack([torch.tensor(np.array(self._postprocess_all2img(x.unsqueeze(0)))) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, cv2.cvtColor(np.vstack(np.asarray(rows)), cv2.COLOR_BGR2RGB))



# ----- Multimodal Datasets ---------
class LANRO(BaseDataset):
    feature_dims = {"front RGB": [64,64,3],
                    "objects": [2,3],
                    "actions": [100,4,1],
                    "language": [4, 16,1],
                    "shapes": [2,6],
                    "colors": [2,6]
                    }  # these feature_dims are also used by the encoder and decoder networks

    def __init__(self, pth, testpth, mod_type):
        super().__init__(pth, testpth, mod_type)
        self.mod_type = mod_type
        self.vocab = self.load_vocab()
        self.feature_dims["language"][1] = len(self.vocab)
        if "level1" in self.path or "level2" in self.path:
            self.feature_dims["language"][0] = 2
        self.vocab_atts = self.load_vocab(atts=True)
        self.lang_labels = None
        self.text2img_size = (64, 250, 3)

    def load_vocab(self, atts=False):
        path = self.path
        vcb = "vocab.txt" if not atts else "vocab_atts.txt"
        if not os.path.exists(os.path.join(os.path.dirname(self.path), vcb)):
            path = os.path.join(os.path.dirname(os.getcwd()), os.path.dirname(self.path), vcb)
        assert os.path.exists(path), "Path to vocab.txt not found"
        vocab = []
        with open(os.path.join(os.path.dirname(path), vcb), "r") as f:
            for line in f:
                vocab.append(line.replace("\n", ""))
        return vocab

    def labels(self):
        if self.current_path is None or self.mod_type != "language":
            return None
        return [" ".join(x.split(" ")[:2]).replace(" the", "") for x in self.lang_labels]

    def _mod_specific_loaders(self):
        return {"front RGB": self.get_rgb, "objects": self.get_objects, "actions": self.get_actions, "language": self.get_lang,
                "colors": self.get_colors, "shapes":self.get_shapes}

    def _mod_specific_savers(self):
        return {"front RGB": self.postprocess_rgb, "objects": self.postprocess_actions, "actions": self.postprocess_actions,
                "language":self.postprocess_language, "colors": self.postprocess_colors, "shapes":self.postprocess_shapes}

    def iter_over_inputs(self, outs, data, mod_names, f=0):
        input_processed = []
        for key, d in data.items():
            if mod_names[key] in ["actions", "objects"]:
                pass
            else:
                output = self._mod_specific_savers()[mod_names[key]](d)
                images = turn_text2image(output, img_size=self.text2img_size) if mod_names[key] in ["language", "colors", "shapes"] \
                    else output #[:, f, :, :, :]
                images = add_recon_title(images, "input\n{}".format(mod_names[key]), (0, 0, 255))
                input_processed.append(np.vstack(images))
                input_processed.append(np.ones((np.vstack(images).shape[0], 2, 3)) * 145)
        if len(input_processed) == 0:
            return None
        inputs = np.hstack(input_processed).astype("uint8")
        return np.hstack((inputs, np.vstack(outs).astype("uint8")))

    def postprocess_rgb(self, data):
        if isinstance(data, dict):
            data = data["data"].reshape(-1, *self.feature_dims["front RGB"])
        else:
            data = data.reshape(-1, *self.feature_dims["front RGB"])
        data = data * 255 if torch.max(data) <= 1 else data
        return np.asarray(data.to(torch.uint8).detach().cpu())

    def postprocess_colors(self, data):
        if isinstance(data, dict):
                o = [seq2words(list(self.vocab_atts), x.detach().cpu()) for x in data["data"]]
                o = [" ".join(x) for i, x in enumerate(o)]
        else:
            o = [" ".join(seq2words(list(self.vocab_atts), x.detach().cpu())) for x in torch.softmax(data, dim=-1)]
        return o

    def postprocess_shapes(self, data):
        if isinstance(data, dict):
                o = [seq2words(list(self.vocab_atts), x.detach().cpu()) for x in data["data"]]
                o = [" ".join(x) for i, x in enumerate(o)]
        else:
            o = [" ".join(seq2words(list(self.vocab_atts), x.detach().cpu())) for x in torch.softmax(data, dim=-1)]
        return o


    def postprocess_language(self, data):
        if isinstance(data, dict):
            data["data"] = torch.argmax(torch.softmax(data["data"].double(), dim=-1), dim=-1)
            o = [([self.vocab[int(round(float(i),0))] for i in x.detach().cpu()]) for x in data["data"]]
            o = [" ".join(list(compress(x, data["masks"][i]))).replace("none","") for i, x in enumerate(o)]
        else:
            o = [" ".join([self.vocab[int(round(float(i),0))] for i in x.detach().cpu()]).replace("none","") for x in torch.argmax(torch.softmax(data, dim=-1), dim=-1)]
        return o

    def postprocess_actions(self, data):
        data = data["data"] if isinstance(data, dict) else data
        return data

    def get_rgb(self):
        data = self.get_data_raw()
        data = torch.stack([torch.tensor(np.asarray(cv2.resize(x,(64,64)))) for x in data])
        data = data.reshape(-1, 3,64,64)
        return data/255

    def get_objects(self):
        data = self.get_data_raw()
        data = [torch.from_numpy(np.asarray(x[0])) for x in data]
        return torch.stack(data)

    def get_shapes(self):
        self.categorical = True
        data = self.get_data_raw()
        d = [one_hot_encode_words(self.vocab_atts, f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in d]
        return torch.stack(data)

    def get_colors(self):
        self.categorical = True
        self.categorical = True
        data = self.get_data_raw()
        d = [one_hot_encode_words(self.vocab_atts, f) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in d]
        return torch.stack(data)

    def get_actions(self):
        self.has_masks = True
        data = self.get_data_raw()
        data = [torch.from_numpy(np.asarray(x)) for x in data]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        data_and_masks = torch.cat((data.reshape(data.shape[0],self.feature_dims[self.mod_type][0], -1), masks), dim=-1)
        return data_and_masks

    def get_lang(self):
        self.has_masks = True
        self.categorical = True
        data = self.get_data_raw()
        self.lang_labels = data
        d = [[self.vocab.index(s) for s in sentence.replace(" object", "").strip().split(" ")] for sentence in data] #[one_hot_encode_words(self.vocab, f.split(" ")) for f in data]
        data = [torch.from_numpy(np.asarray(x)) for x in d]
        masks = lengths_to_mask(torch.tensor(np.asarray([x.shape[0] for x in data]))).unsqueeze(-1)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
        data = torch.nn.functional.one_hot(data)
        data_and_masks = torch.cat((data, masks), dim=-1)
        return data_and_masks

    def _postprocess_all2img(self, data):
        """
        Converts any kind of data to images to save traversal visualizations

        :param data: input data
        :type data: torch.tensor
        :return: processed images
        :rtype: torch.tensor
        """
        output_processed = self._postprocess(data)
        output_processed = turn_text2image(output_processed, img_size=self.text2img_size) \
            if self.mod_type in ["language", "colors", "shapes"] else output_processed
        return output_processed

    def save_recons(self, data, recons, path, mod_names):
        if self.mod_type == "language" and [i for i in mod_names if mod_names[i]=="language"][0] in data.keys():
            recons = {"data":recons, "masks":data[[i for i in mod_names if mod_names[i]=="language"][0]]["masks"]}
        output_processed = self._postprocess_all2img(recons)
        rgb_mods = [k for k, v in mod_names.items() if "RGB" in v]
        if self.mod_type not in ["actions", "objects"]:
            outs = add_recon_title(output_processed, "output\n{}".format(self.mod_type), (0, 170, 0))
            final = self.iter_over_inputs(outs, data, mod_names)
            if final is not None:
                cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        else:
            actions = list(output_processed)
            d = {"actions": actions}
            for i, md in enumerate(["mod_1", "mod_2", "mod_3"]):
                gt = self._mod_specific_savers()[mod_names[md]](data[md]) if md in data.keys() else None
                d["{}_gt".format(mod_names[md])] = gt
            with open(path.replace("png", "pkl"), 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def save_traversals(self, recons, path, num_dims):
        """
        Makes a grid of traversals and saves as animated gif image

        :param recons: data to save
        :type recons: torch.tensor
        :param path: path to save the traversal to
        :type path: str
        :param num_dims: number of latent dimensions
        :type num_dims: int
        """
        if len(recons.shape) < 5:
            output_processed = torch.tensor(np.asarray(self._postprocess_all2img(recons))).transpose(1, 3)
            grid = np.asarray(make_grid(output_processed, padding=1, nrow=int(math.sqrt(len(recons)))).transpose(2, 0))
            cv2.imwrite(path, grid.astype("uint8"))
        else:
            output_processed = torch.stack([torch.tensor(self._postprocess_all2img(x)) for x in recons])
            output_processed = output_processed.reshape(num_dims, -1, *output_processed.shape[1:]).squeeze()
            rows = []
            for ind, dim in enumerate(output_processed):
                rows.append(np.asarray(torch.hstack([x for x in dim]).type(torch.uint8).detach().cpu()))
            cv2.imwrite(path, np.vstack(np.asarray(rows)))
