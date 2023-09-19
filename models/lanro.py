"""A script to demonstrate grasping."""
import gymnasium as gym
import lanro_gym
import time as time
import numpy as np
import math
import pickle
import cv2


env = gym.make("PandaNLRight2Shape-v0", render=False)
total_ep = 100
start_t = time.time()
obj_poses = []
instructions = []
robot_poses = []
object_colors = []
object_shapes = []
images = []
correct_objects = []
while not len(instructions) == total_ep:
    print(len(instructions))
    obs, info = env.reset()
    goal_pos = env.sim.get_base_position(env.task.goal_object_body_key)
    instruction = env.env.env.env.decode_instruction(obs["instruction"])
    print(instruction)
    obj_pos = []
    robot_pos = []
    shape_lib = []
    col_lib = []
    image = env.env.env.env.robot.get_camera_img()
    correct_object = None
    #cv2.imwrite("limg{}.png".format(len(instructions)), image)
    for x in [x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]:
        col_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_color().name)
        shape_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_shape().name)
    for i in range(env._max_episode_steps * 2):
        oo = []
        for idx, x in enumerate([x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]):
            oo.append(list(env.env.env.env.sim.get_base_position(x)))
            if x == env.task.goal_object_body_key:
                correct_object = idx
        obj_pos.append(oo)
        ee_pos = obs['observation'][:3]
        if i < 35:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = 1
        elif i < 45:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = -1
        elif i < 60:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[1] = action[1] + 2
            action[2] = action[2] + 0.03
            action[3] = -1
        env.step(action)
        robot_pos.append([*env.env.env.env.robot.get_current_pos(), action[3]])
    if math.dist(goal_pos, env.sim.get_base_position(env.task.goal_object_body_key)) > 0.07:
        images.append(image)
        instructions.append(instruction)
        obj_poses.append(list(obj_pos))
        robot_poses.append(robot_pos)
        object_colors.append(col_lib)
        object_shapes.append(shape_lib)
        correct_objects.append(correct_object)
    else:
        print("FAIL")
env.close()
print(correct_objects)
env = gym.make("PandaNLLeft2Shape-v0", render=False)
start_t = time.time()
sofar = len(instructions)
while not len(instructions) == (total_ep + sofar):
    print(len(instructions))
    obs, info = env.reset()
    goal_pos = env.sim.get_base_position(env.task.goal_object_body_key)
    instruction = env.env.env.env.decode_instruction(obs["instruction"])
    print(instruction)
    obj_pos = []
    robot_pos = []
    shape_lib = []
    col_lib = []
    image = env.env.env.env.robot.get_camera_img()
    correct_object = None
    for x in [x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]:
        col_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_color().name)
        shape_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_shape().name)
    for i in range(env._max_episode_steps * 2):
        oo = []
        for idx, x in enumerate([x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]):
            oo.append(list(env.env.env.env.sim.get_base_position(x)))
            if x == env.task.goal_object_body_key:
                correct_object = idx
        obj_pos.append(oo)
        ee_pos = obs['observation'][:3]
        if i < 35:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = 1
        elif i < 45:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = -1
        elif i < 60:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[1] = action[1] - 2
            action[2] = action[2] + 0.03
            action[3] = -1
        env.step(action)
        robot_pos.append([*env.env.env.env.robot.get_current_pos(), action[3]])
    if math.dist(goal_pos, env.sim.get_base_position(env.task.goal_object_body_key)) > 0.07:
        images.append(image)
        instructions.append(instruction)
        obj_poses.append(list(obj_pos))
        robot_poses.append(robot_pos)
        object_colors.append(col_lib)
        object_shapes.append(shape_lib)
        correct_objects.append(correct_object)
    else:
        print("FAIL")
env.close()

env = gym.make("PandaNLLift2Shape-v0", render=False)
start_t = time.time()
sofar = len(instructions)
while not len(instructions) == (total_ep + sofar):
    print(len(instructions))
    obs, info = env.reset()
    goal_pos = env.sim.get_base_position(env.task.goal_object_body_key)
    instruction = env.env.env.env.decode_instruction(obs["instruction"])
    print(instruction)
    obj_pos = []
    robot_pos = []
    shape_lib = []
    col_lib = []
    image = env.env.env.env.robot.get_camera_img()
    correct_object = None
    for x in [x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]:
        col_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_color().name)
        shape_lib.append(env.env.env.env.task.task_object_list.objects[int(x.split("object")[-1])].get_shape().name)
    for i in range(env._max_episode_steps * 2):
        oo = []
        for idx, x in enumerate([x for x in env.env.env.env.sim._bodies_idx.keys() if "object" in x]):
            oo.append(list(env.env.env.env.sim.get_base_position(x)))
            if x == env.task.goal_object_body_key:
                correct_object = idx
        obj_pos.append(oo)
        ee_pos = obs['observation'][:3]
        if i < 35:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = 1
        elif i < 45:
            action = np.concatenate((goal_pos - ee_pos, [0]))
            action[3] = -1
        elif i < 60:
            action = np.zeros((4, ))
            action[2] = 0.05
            action[3] = -1
        env.step(action)
        robot_pos.append([*env.env.env.env.robot.get_current_pos(), action[3]])
    if math.dist(goal_pos, env.sim.get_base_position(env.task.goal_object_body_key)) > 0.07:
        images.append(image)
        instructions.append(instruction)
        obj_poses.append(list(obj_pos))
        robot_poses.append(robot_pos)
        object_colors.append(col_lib)
        object_shapes.append(shape_lib)
        correct_objects.append(correct_object)
    else:
        print("FAIL")
with open('lanro_robot_poses.pkl', 'wb') as handle:
    pickle.dump(robot_poses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lanro_object_poses.pkl', 'wb') as handle:
    pickle.dump(obj_poses, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lanro_instructions.pkl', 'wb') as handle:
    pickle.dump(instructions, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lanro_colors.pkl', 'wb') as handle:
    pickle.dump(object_colors, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lanro_shapes.pkl', 'wb') as handle:
    pickle.dump(object_shapes, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lanro_images.pkl', 'wb') as handle:
    pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('correct_objects.pkl', 'wb') as handle:
    pickle.dump(correct_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)