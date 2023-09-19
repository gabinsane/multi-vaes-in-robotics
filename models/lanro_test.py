
import gymnasium as gym
import lanro_gym
import pickle, math
import argparse
import numpy as np
import cv2, os
import glob
from eval.infer import MultimodalVAEInfer
import copy
import time, os
from models.datasets import LANRO
from typing import List
import pybullet as p

def scale_rgb(rgb_lst: List[float]) -> List[float]:
    return [_color / 255.0 for _color in rgb_lst]

def colormap(color):
    map = {"RED": scale_rgb([255.0, 87.0, 89.0]), "BLUE":scale_rgb([78.0, 121.0, 167.0]), "GREEN":scale_rgb([89.0, 169.0, 79.0]), "YELLOW":scale_rgb([255,255,0])}
    return map[color]

def make_object(shape, pos, name):
    if shape.lower() == "soap":
        env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/soap.urdf",
                               basePosition=[0, 0, 0.1])
    elif shape.lower() == "mug":
        env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/mug.urdf"
                               )
    elif shape.lower() == "lemon":
       env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/lemon.urdf")
    elif shape.lower() == "toothpaste":
        env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/toothpaste.urdf")
    elif shape.lower() == "stapler":
        env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/stapler.urdf")
    elif shape.lower() == "teabox":
        env.env.env.env.sim.loadURDF(body_name=name,
                               fileName="./models/lanro_gym/objects_urdfs/teabox.urdf")
    env.env.env.env.sim.set_base_pose(name, pos, [0, 0, 0, 1])


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
    mods[get_mod_idx("actions")]["data"] = None
    mods[get_mod_idx("actions")]["masks"] = None
    correct_objects = load_pickle(correct_objects)[::subsample]
    joints_gt = load_pickle(joints)[::subsample]
    return mods, data, test_shapes, test_objects, correct_objects, joints_gt

def spawn_objects(objs, idx):
    num_objs = [0, 1, 1, 4, 4]
    for x in range(num_objs[level]):
        env.env.env.env.sim.remove_body("o{}".format(x))
    obs, info = env.reset()
    for x in range(num_objs[level]):
        make_object(shapes[idx][x], objs[x], "o{}".format(x))

def get_gripper_pos():
    return env.env.env.env.robot.sim.bclient.getLinkState(0, 11)[0] + \
        env.env.env.env.robot.sim.bclient.getLinkState(0, 11)[1]

def get_obj_index(obj_name):
    return env.env.env.env.sim.get_object_id(obj_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model", type=str, help="path to the .ckpt model file. Relative or absolute")
    parser.add_argument("-s", "--subsample", type=int, help="use every nth trial of the 1200 test trials", default=1)
    parser.add_argument("-d", "--dataset", type=int, help="number of the dataset used for training (1-4)")
    args = parser.parse_args()
    model_ckpts = [args.model]
    for model_ckpt in model_ckpts:
        print(model_ckpt)
        try:
            model_cfg = (os.path.dirname(os.path.dirname(model_ckpt)) + "/config.yml")
            testset = "testset"
            subsample = args.subsample
            level = args.dataset
            render = False
            show_ground_truth = False

            levels = ["lanro_ycb/level1/", "lanro_ycb/level2/", "lanro_ycb/level3/", "lanro_ycb/"]
            pt = "./data/{}".format(levels[level - 1])
            infer_model = MultimodalVAEInfer(model_ckpt, model_cfg)
            #data, labels = infer_model.datamodule.get_num_samples(10, split="val")
            mods, data, test_shapes, test_objects, correct_objects, joints_gt = load_testdata()
            cor_obs_cats = {"mug":0, "toothpaste":0, "stapler":0, "lemon":0, "teabox":0, "soap":0}
            cor_actions = {"lift": 0, "left":0, "right":0}
            cor_obs_toothpaste = []
            correctly_selected_objects = []
            correctly_moved = []
            correctly_moved_and_selected = []
            failed_gt = []
            env = gym.make("PandaNLReach2-v0", action_type= "end_effector", render=render)
            env.test_only = True
            # inference
            r = int(1200/(subsample*10)) if subsample <= 5 else 1
            actions_gt = postprocess_data(data, get_mod_idx("actions"))
            language_gt = postprocess_data(data, get_mod_idx("language"))
            # objects_gt = postprocess_data(data, get_mod_idx("objects"))
            objects_gt = load_pickle(test_objects)[::subsample]
            shapes = load_pickle(test_shapes)[::subsample]
            rgb_gt = postprocess_data(data, get_mod_idx("front RGB"))
            for batch in range(r):
                print("BATCH", batch)
                if subsample > 5:
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
                actions = process_model_output(output, "actions")
                for i, trial in enumerate(actions):
                    #print(i)
                    idx = 0 if subsample > 5 else (batch*10)
                    sel = 0
                    correct_object = ["o0", "o1", "o2", "o3", "o4"][correct_objects[idx+i]]
                    remove_obj_keys = [key for key in env.sim._bodies_idx.keys() if 'object' in key]
                    for _key in remove_obj_keys:
                         env.sim.remove_body(_key)
                    spawn_objects(objects_gt[idx+i][0], idx+i)
                    # im = cv2.cvtColor(rgb_gt[i], cv2.COLOR_BGR2RGB)
                    # cv2.imshow("SCENE", im)
                    # cv2.waitKey(3)
                    env.env.env.env.sim.bclient.addUserDebugText(language_gt[idx+i], [0.05, -.3, .4], textSize=2.0, replaceItemUniqueId=43)
                    if show_ground_truth:
                        env.env.env.env.sim.bclient.addUserDebugText("ground truth" , [0.05, 0.3, .6], textSize=2.0, replaceItemUniqueId=53)
                        if actions_gt is not None:
                            for action in joints_gt[idx+i]:
                                aa = list(action.squeeze().detach().cpu())
                                ac = np.asarray(aa)
                                env.step(ac)
                            spawn_objects(objects_gt[idx+i][0], idx+i)
                    magnetized = None
                    env.env.env.env.sim.bclient.addUserDebugText("inference" , [0.05, 0.3, .6], textSize=2.0, replaceItemUniqueId=53)
                    for action in trial:
                        aa = list(action.squeeze())
                        ac = np.asarray(aa)
                        env.step(ac)
                        if get_gripper_pos()[:3][-1] <= 0.2 and magnetized is None:
                            for o in ["o0", "o1", "o2", "o3"][:np.max(correct_objects)+1]:
                                if np.linalg.norm(
                                        np.asarray(get_gripper_pos()[:3]) - np.asarray(env.env.env.env.sim.get_base_position(o))) <= 0.10:
                                    if o == correct_object:
                                        sel = 1
                                        if level > 2:
                                            cor_obs_cats[language_gt[idx+i].split(" ")[-1]] += 1
                                        # posg = list(get_gripper_pos()[:3])
                                        # posg[-1] = posg[-1] - 0.03
                                        # env.env.env.env.sim.set_base_pose(o, posg, [0, 0, 0, 1])
                        if magnetized is not None:
                                p.changeConstraint(magnetized, get_gripper_pos()[:3], get_gripper_pos()[3:])
                    if "left" in language_gt[idx+i]:
                        if (env.env.env.env.sim.get_base_position(correct_object)[1] -
                                     objects_gt[idx+i][0][int(correct_object[-1]) - 1][1]) < -0.3:
                            correctly_moved.append(1)
                            #print("CORRECT LEFT")
                            c = 1
                            cor_actions["left"] += 1
                        else:
                            c = 0
                            correctly_moved.append(0)
                    elif "right" in language_gt[idx+i]:
                        if (env.env.env.env.sim.get_base_position(correct_object)[1] -
                                     objects_gt[idx+i][0][int(correct_object[-1]) - 1][1]) > 0.3:
                            correctly_moved.append(1)
                            #print("CORRECT RIGHT")
                            cor_actions["right"] += 1
                            c = 1
                        else:
                            c = 0
                            correctly_moved.append(0)
                    else:
                        t1 =  (env.env.env.env.sim.get_base_position(correct_object)[1] -
                                     objects_gt[idx+i][0][int(correct_object[-1]) - 1][1]) > 0.3
                        t2 = (env.env.env.env.sim.get_base_position(correct_object)[1] -
                                     objects_gt[idx+i][0][int(correct_object[-1]) - 1][1]) < -0.3
                        if (env.env.env.env.sim.get_base_position(correct_object)[2] -
                                     objects_gt[idx+i][0][int(correct_object[-1]) - 1][2]) > 0.04 and not t1 and not t2:
                            correctly_moved.append(1)
                            # print(env.env.env.env.sim.get_base_position(correct_object)[1])
                            # print(objects_gt[i][0][int(correct_object[-1]) - 1][1])
                            #print("CORRECT LIFT")
                            cor_actions["lift"] += 1
                            c = 1
                        else:
                            correctly_moved.append(0)
                            c = 0
                    correctly_selected_objects.append(sel)
                    #if sel == 1:
                        #print("Correctly selected", o)
                    if sel == 1 and c == 1:
                        correctly_moved_and_selected.append(1)
                    else:
                        correctly_moved_and_selected.append(0)
                    if batch % 10 == 0 and batch > 0:
                        env.close()
                        env = gym.make("PandaNLReach2-v0", action_type="end_effector", render=render)
                        env.test_only = True
            env.close()
            print(model_ckpt)
            print("Correctly selected:")
            print("{}/{}, that is {}%".format(sum(correctly_selected_objects),len(correctly_selected_objects), sum(correctly_selected_objects)*100/len(correctly_selected_objects)))
            print("Correctly moved:")
            print("{}/{}, that is {}%".format(sum(correctly_moved), len(correctly_moved), sum(correctly_moved)*100/len(correctly_moved)))
            print("Correctly moved and selected:")
            print("{}/{}, that is {}%".format(sum(correctly_moved_and_selected), len(correctly_moved_and_selected), sum(correctly_moved_and_selected)*100/len(correctly_moved_and_selected)))
            print(cor_obs_cats)
            n_a = int(len(correctly_selected_objects)/ 3) if level != 3 else len(correctly_selected_objects)
            if level != 3:
                print("lift: {} move left: {} move right: {}".format(cor_actions["lift"]/n_a, cor_actions["left"]/n_a, cor_actions["right"]/n_a))
            if len(correctly_selected_objects) > 100:
                with open(os.path.join(os.path.dirname(model_ckpt), 'stats_no_magnet.txt'), 'w') as f:
                        f.write("Correctly selected: \n")
                        f.write("{}/{}, that is {}% \n".format(sum(correctly_selected_objects),len(correctly_selected_objects), sum(correctly_selected_objects)*100/len(correctly_selected_objects)))
                        f.write("Correctly moved:\n")
                        f.write("{}/{}, that is {}%\n".format(sum(correctly_moved), len(correctly_moved), sum(correctly_moved)*100/len(correctly_moved)))
                        f.write("Correctly moved and selected:\n")
                        f.write("{}/{}, that is {}%\n".format(sum(correctly_moved_and_selected), len(correctly_moved_and_selected), sum(correctly_moved_and_selected)*100/len(correctly_moved_and_selected)))
                        if level > 2:
                            f.write("Objects:\n")
                            f.write("mug: {} toothpaste: {} stapler: {} lemon: {} teabox: {} soap: {}\n".format(cor_obs_cats["mug"], cor_obs_cats["toothpaste"], cor_obs_cats["stapler"], cor_obs_cats["lemon"], cor_obs_cats["teabox"], cor_obs_cats["soap"]))
                        if level != 3:
                            f.write("Actions:\n")
                            f.write("lift: {} move left: {} move right: {}".format(cor_actions["lift"]/n_a, cor_actions["left"]/n_a, cor_actions["right"]/n_a))
        except NameError as e:
            print(str(e))
            print(model_ckpt)
        except RuntimeError as e:
            print(str(e))
            print(model_ckpt)
