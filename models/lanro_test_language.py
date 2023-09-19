import pickle, math
import numpy as np
import cv2, os
import glob
import torch
import argparse
from eval.infer import MultimodalVAEInfer
import copy
import time, os
from models.datasets import LANRO
import pybullet as p

def load_pickle(path):
  with open(path, 'rb') as handle:
    data = pickle.load(handle)
  return data


def process_model_output(output, mod_name):
    return output.mods[get_mod_idx(mod_name)].decoder_dist.loc.detach().cpu().numpy()


def postprocess_data(data, mod):
    return infer_model.datamod.datasets[int(mod.split("_")[-1])-1]._postprocess(data[mod])


def get_mod_idx(name):
    for key, item in infer_model.model.mod_names.items():
        if item == name:
            return key

def cast_dict_to_cuda(d):
    for key in d.keys():
        if d[key]["data"] is not None:
            d[key]["data"] = d[key]["data"].cuda()
        if d[key]["masks"] is not None:
            d[key]["masks"] = d[key]["masks"].cuda()
    return d

def load_testdata():
    test_actions = "{}{}/lanro_robot_ees.pkl".format(pt, testset)
    joints = "{}{}/lanro_actions.pkl".format(pt, testset)
    test_instructions = "{}{}/lanro_instructions.pkl".format(pt, testset)
    test_objects = "{}{}/lanro_object_poses.pkl".format(pt, testset)
    test_images = "{}{}/lanro_images.pkl".format(pt, testset)
    test_colors = "{}{}/lanro_colors.pkl".format(pt, testset)
    test_shapes = "{}{}/lanro_shapes.pkl".format(pt, testset)
    correct_objects = "{}{}/correct_objects.pkl".format(pt,testset)
    data_loaders = [LANRO(test_instructions, None, "language"), LANRO(test_actions, None, "actions"),  LANRO(test_images, None, "front RGB")]
    data = {}
    for idx, loader in enumerate(data_loaders):
        data["mod_{}".format(idx + 1)] = {}
        d = loader.get_data()[::subsample]
        data["mod_{}".format(idx+1)]["data"] = d
        data["mod_{}".format(idx + 1)]["masks"] = None
        if loader.has_masks:
            if len(d.shape) == 3:
                data["mod_{}".format(idx + 1)] = {"data": d[:,:,:-1], "masks": d[:,:,-1].bool()}
            elif len(d.shape) == 4:
                data["mod_{}".format(idx + 1)] = {"data": d[:,:,:, :-1], "masks": d[:,:,0,-1].squeeze().bool()}
    mods = copy.deepcopy(data)
    mods[get_mod_idx("language")]["data"] = None
    #mods[get_mod_idx("language")]["masks"] = None
    correct_objects = load_pickle(correct_objects)[::subsample]
    joints_gt = load_pickle(joints)[::subsample]
    return mods, data, test_shapes, test_objects, correct_objects, joints_gt

def replace_iter(text, synonyms, correct_word):
    for s in synonyms:
        text = text.replace(s, correct_word)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-s", "--subsample", type=int, help="use every nth trial of the 1200 test trials", default=1)
    parser.add_argument("-d", "--dataset", type=int, help="number of the dataset used for training (1-4)")
    args = parser.parse_args()
    model_ckpts = [args.model]
    for model_ckpt in model_ckpts:
        for model_ckpt in model_ckpts:
        try:
            if not os.path.exists(model_ckpt.replace("last.ckpt", "stats_language.txt")):
                model_cfg = (os.path.dirname(os.path.dirname(model_ckpt)) + "/config.yml")
                testset = "testset"
                subsample = args.subsample
                level = args.dataset
                levels = ["lanro_ycb/level1/", "lanro_ycb/level2/", "lanro_ycb/level3/", "lanro_ycb/"]
                pt = "./data/{}".format(levels[level - 1])
                infer_model = MultimodalVAEInfer(model_ckpt, model_cfg)
                mods, data, test_shapes, test_objects, correct_objects, joints_gt = load_testdata()
                correct_action = []
                correct_object = []
                correct_absolute = []
                # inference
                r = 120 if subsample == 1 else 1
                actions_gt = postprocess_data(data, get_mod_idx("actions"))
                language_gt = postprocess_data(data, get_mod_idx("language"))
                # objects_gt = postprocess_data(data, get_mod_idx("objects"))
                objects_gt = load_pickle(test_objects)[::subsample]
                shapes = load_pickle(test_shapes)[::subsample]
                rgb_gt = postprocess_data(data, get_mod_idx("front RGB"))
                for batch in range(r):
                    #print("BATCH", batch)
                    if subsample > 1:
                        mods_batch = mods
                    else:
                        mods_batch = {}
                        for k, val in mods.items():
                            mods_batch[k] = {"data": None, "masks":None}
                            if val["data"] is not None:
                                mods_batch[k]["data"] = val["data"][batch * 10:(batch * 10 + 10)]
                            if val["masks"] is not None:
                                mods_batch[k]["masks"] = val["masks"][batch*10:(batch*10+10)]
                    output = infer_model.model.model.forward(cast_dict_to_cuda(mods_batch))
                    language_recon = infer_model.datamod.datasets[0].postprocess_language(torch.tensor(process_model_output(output, "language")))
                    for i, sentence in enumerate(language_recon):
                        idx = 0 if subsample > 1 else (batch * 10)
                        sentence = replace_iter(sentence.strip(), ["hoist", "raise"], "lift")
                        sentence = replace_iter(sentence, ["drag", "shift"], "move")
                        gt = language_gt[idx + i]
                        gt = replace_iter(gt, ["hoist", "raise"], "lift")
                        gt = replace_iter(gt, ["drag", "shift"], "move")
                        absolute = int(gt == sentence)
                        correct_absolute.append(absolute)
                        if level > 2:
                            obj = int(gt.split(" ")[-1] == sentence.split(" ")[-1])
                            if sentence.split(" ")[-1] == "":
                                obj = 0
                            correct_object.append(obj)
                        if level != 3:
                            if "left" in gt or "right" in gt:
                                action = int(gt.split(" ")[:2] == sentence.split(" ")[:2])
                                if "" in sentence.split(" ")[:2]:
                                    action = 0
                            else:
                                action = int(gt.split(" ")[0] == sentence.split(" ")[0])
                                if sentence.split(" ")[0] == "":
                                    action = 0
                            correct_action.append(action)
                print(model_ckpt)
                if level > 2:
                    print("Correct object:")
                    print("{}/{}, that is {}%".format(sum(correct_object),len(correct_object), sum(correct_object)*100/len(correct_object)))
                if level != 3:
                    print("Correct action:")
                    print("{}/{}, that is {}%".format(sum(correct_action), len(correct_action), sum(correct_action)*100/len(correct_action)))
                print("Correct absolute:")
                print("{}/{}, that is {}%".format(sum(correct_absolute), len(correct_absolute), sum(correct_absolute)*100/len(correct_absolute)))
                if len(correct_absolute) > 100:
                    with open(os.path.join(os.path.dirname(model_ckpt), 'stats_language.txt'), 'w') as f:
                        if level > 2:
                                f.write("Correct object: \n")
                                f.write("{}/{}, that is {}%".format(sum(correct_object),len(correct_object), sum(correct_object)*100/len(correct_object)))
                        if level != 3:
                            f.write("Correct action:\n")
                            f.write("{}/{}, that is {}%".format(sum(correct_action), len(correct_action), sum(correct_action)*100/len(correct_action)))
                        f.write("Correct absolute:\n")
                        f.write("{}/{}, that is {}%".format(sum(correct_absolute), len(correct_absolute), sum(correct_absolute)*100/len(correct_absolute)))
        except NameError as e:
            print(str(e))
            print(model_ckpt)
        except RuntimeError as e:
            print(str(e))
            print(model_ckpt)
