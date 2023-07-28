from enum import Enum
from ai2thor.controller import Controller
import matplotlib.pyplot as plt
from PIL import Image
from gpt_chat import gpt
import time
import re
import numpy as np
import json
import math
import os


DIRECTION_DICT = {
    "Front": 0,
    "Front Right": 45,
    "Right": 90,
    "Rear Right": 135,
    "Rear": 180,
    "Rear Left": 225,
    "Left": 270,
    "Front Left": 315}

#### NONE CLASS simulator funcions ####


def get_vaild_obj_types(event):
    rel_objs = event.metadata['objects']
    vaild_obj_types = []

    for each in rel_objs:
        if each["objectType"] not in vaild_obj_types:
            vaild_obj_types.append(each["objectType"])

    print(len(vaild_obj_types))
    print(vaild_obj_types)
    return vaild_obj_types

# 计算物体相对相机的角度


def calculate_orientation(image_width, hFOV, x):
    # Calculate the pixel distance from the center of the image
    d = x - image_width / 2
    # Calculate the proportion of this distance to the total width of the image
    proportion = d / (image_width / 2)
    # Calculate the orientation in degrees
    orientation = proportion * (hFOV / 2)

    return orientation


def get_orientation_str(image_width, hFOV, x):
    orientation = calculate_orientation(image_width, hFOV, x)
    unit = hFOV / 6
    if orientation > unit:
        return "right"
    elif orientation < -unit:
        return "left"
    else:
        return "center"

# 规范化坐标点（四舍五入，保留3位小数）


def float_point_regularization(point):
    return (round(point[0], 3), round(point[1], 3))


def check_point_in_valid_list(point, valid_points):
    for each in valid_points:
        if abs(each[0] - point[0]) < 0.1 and abs(each[1] - point[1]) < 0.1:
            return True
    return False

# 根据网格和有效点位置，计算出非有效点的列表


def get_reach_valid_points(d, valid_points):
    valid_points = [(round(x, 3), round(z, 3)) for (x, z) in valid_points]
    # for each in valid_points:
    #     print(f"x:{each[0]} , z:{each[1]}")

    # 将点的坐标拆分为两个列表
    x_valid = [point[0] for point in valid_points]
    z_valid = [point[1] for point in valid_points]

    # 获取x和z的最小和最大值以定义网格
    x_min, x_max = min(x_valid), max(x_valid)
    z_min, z_max = min(z_valid), max(z_valid)

    # 创建网格
    x = np.arange(x_min, x_max + 2*d, d)
    z = np.arange(z_min, z_max + 2*d, d)
    X, Z = np.meshgrid(x, z)

    # 将网格点的坐标拆分为两个列表
    x_all = X.ravel()
    z_all = Z.ravel()

    # 筛选出非有效点
    invalid_points = [(x, z) for x, z in zip(x_all, z_all)
                      if not check_point_in_valid_list((x, z), valid_points)]

    return valid_points, invalid_points


# return controller, init_event
# 根据数据集初始化场景，传送到起始位置
def start_scene_by_episode(sim_args, episode):
    # init
    controller = Controller(
        agentMode="locobot",
        visibilityDistance=sim_args.VISIBLE_DISTANCE,
        gridSize=sim_args.GRID_SIZE,
        width=sim_args.VIEW_WIDTH,
        height=sim_args.VIEW_HEIGHT,
        fieldOfView=sim_args.FIELD_OF_VIEW,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        scene=episode['scene'])

    target_object_type = episode['object_type']
    # print("target type: " + target_object_type)

    # teleport to start pos
    init_event = controller.step(
        action="Teleport",
        position=episode['initial_position'],
        rotation={"x": 0, "y": episode["initial_orientation"], "z": 0},
        horizon=episode["initial_horizon"]
    )

    # print(init_event.metadata['agent'])

    # get vaild object type list
    # self.vaild_obj_types = get_vaild_obj_types(init_event)
    return controller, init_event, target_object_type


def reset_scene_by_episode(controller, sim_args, episode):
    controller.reset(visibilityDistance=sim_args.VISIBLE_DISTANCE,
                     gridSize=sim_args.GRID_SIZE,
                     width=sim_args.VIEW_WIDTH,
                     height=sim_args.VIEW_HEIGHT,
                     fieldOfView=sim_args.FIELD_OF_VIEW,
                     scene=episode['scene'])

    target_object_type = episode['object_type']

    # teleport to start pos
    init_event = controller.step(
        action="Teleport",
        position=episode['initial_position'],
        rotation={"x": 0, "y": episode["initial_orientation"], "z": 0},
        horizon=episode["initial_horizon"]
    )

    return init_event, target_object_type

# 获得当前视角所有可识别物体的名称


def get_scene_objs(sim_args, detections, depth_frame):
    scene_objs = []
    for key in detections.keys():
        value = detections[key]
        center_x = (value[0] + value[2]) / 2
        center_y = (value[1] + value[3]) / 2
        depth_distance = depth_frame[int(center_y)][int(center_x)]

        # add scene obj data
        obj_name = str(key).split("|")[0]
        # fliters
        if sim_args.USE_VAILD_OBJ_TYPES:
            if obj_name not in sim_args.vaild_obj_types:
                continue

        # fix system name of border objects.
        if "wall" in obj_name:
            obj_name = "Wall"

        new_obj = obj_name
        scene_objs.append(new_obj)

    return scene_objs

# 执行旋转环顾四周


def get_observation(controller, sim_args, deg_unit):

    deg = deg_unit
    observations = []
    summarize_observations = []
    for i in range(int(360/deg)):
        # 先获取观察数据，再旋转
        data, summarize_data = get_direction_objs(
            i, controller, sim_args, deg_unit)
        observations.append(data)
        summarize_observations.append(summarize_data)
        controller.step("RotateRight", degrees=deg)

    # json_data = json.dumps(observations)
    # print("Return: \n" + json_data)
    return observations, summarize_observations


# 获得一个朝向的俯仰角可见物体json格式观察结果


def get_json_objects_result(controller, sim_args, is_check_front_target=False):
    objs = {}

    for horizon_deg in (0, -30, 30):
        # 0 middle -30 up 30 down
        event = controller.step(
            action="Teleport",
            horizon=horizon_deg
        )

        depth_frame = event.depth_frame
        detections = event.instance_detections2D

        get_objects_around(sim_args, objs, horizon_deg,
                           detections, depth_frame)

    if len(objs) <= 0:
        return None

    res_objs_list = []
    for name in objs.keys():
        orientation_str = " and ".join([part for part in objs[name][0]])
        distance = objs[name][1]
        if distance > sim_args.VISIBLE_DISTANCE:
            distance_str = f"{distance:.2f}m[TOO FAR]"
        else:
            distance_str = f"{distance:.2f}m"
        new_obj = {name: "%s, %s" % (orientation_str, distance_str)}

        res_objs_list.append(new_obj)

    if is_check_front_target:
        if sim_args.target_object_type in objs.keys():
            distance = objs[sim_args.target_object_type][1]
            if distance <= sim_args.VISIBLE_DISTANCE:
                sim_args.rules_achieve = True
            else:
                sim_args.rules_achieve = False

    # 俯仰角复位
    event = controller.step(
        action="Teleport",
        horizon=0
    )
    json_data = json.dumps(res_objs_list)
    return json_data

# 获得可视物体详情(指定俯仰角度)


def get_objects_around(sim_args, obj_dist, horizon_deg, detections, depth_frame):
    DICT_HORIZON_DEG = {
        0: "Middle",
        -30: "Top",
        30: "Down"
    }

    for key in detections.keys():
        # get raw image data
        value = detections[key]
        center_x = (value[0] + value[2]) / 2
        center_y = (value[1] + value[3]) / 2
        depth_distance = depth_frame[int(center_y)][int(center_x)]

        # calc scene obj data
        obj_name = str(key).split("|")[0]
        distance = float(depth_distance)
        orient_part_str = DICT_HORIZON_DEG[horizon_deg] + " " + get_orientation_str(
            sim_args.VIEW_WIDTH, sim_args.FIELD_OF_VIEW, center_x)

        # fliters
        if sim_args.USE_VAILD_OBJ_TYPES:
            if obj_name not in sim_args.vaild_obj_types:
                continue
        # fix system name of border objects.
        if "wall" in obj_name:
            obj_name = "Wall"

        if obj_name in obj_dist.keys():
            obj_dist[obj_name][0].append(orient_part_str)
        else:
            obj_dist[obj_name] = ([orient_part_str], distance)

# 通过api获得所有可移动到的点位


def get_grid_observation_points(controller, deg_unit):

    get_RP_event = controller.step(action="GetReachablePositions")
    rps = get_RP_event.metadata['actionReturn']
    agent_metadata = get_RP_event.metadata['agent']

    agent_pos = agent_metadata['position']
    agent_p = (agent_pos['x'], agent_pos['z'])

    degrees = int(
        round(agent_metadata['rotation']['y'] / deg_unit) * deg_unit)
    print(f"[degrees]: {degrees}")
    # d_rad = np.radians(degrees)
    # dir_vector = np.array([np.sin(d_rad), 0, np.cos(d_rad)])

    vaild_p = []
    for each in rps:
        vaild_p.append((each['x'], each['z']))

    return agent_p, vaild_p, degrees

# 获得当前朝向最大可移动距离


def get_max_access_distance(controller, sim_args, deg_unit):
    GRID_SIZE = sim_args.GRID_SIZE
    current_point, valid_points, w_degrees = get_grid_observation_points(
        controller, deg_unit)
    # 0度面朝上，向右转为正
    dir_dict = {0: (0, GRID_SIZE),
                360: (0, GRID_SIZE),
                45: (GRID_SIZE, GRID_SIZE),
                90: (GRID_SIZE, 0),
                135: (GRID_SIZE, -GRID_SIZE),
                180: (0, -GRID_SIZE),
                225: (-GRID_SIZE, -GRID_SIZE),
                270: (-GRID_SIZE, 0),
                315: (-GRID_SIZE, GRID_SIZE)}
    # 0度面朝下，向右转为正
    # dir_dict = {0: (0, -GRID_SIZE),
    #             360: (0, -GRID_SIZE),
    #             45: (-GRID_SIZE, -GRID_SIZE),
    #             90: (-GRID_SIZE, 0),
    #             135: (-GRID_SIZE, GRID_SIZE),
    #             180: (0, GRID_SIZE),
    #             225: (GRID_SIZE, GRID_SIZE),
    #             270: (GRID_SIZE, 0),
    #             315: (GRID_SIZE, -GRID_SIZE)}

    # 有效点规范化
    current_point = float_point_regularization(current_point)
    valid_points, invalid_points = get_reach_valid_points(
        GRID_SIZE, valid_points)
    # print(f"degree: {w_degrees}, valid_points:")
    # print(valid_points)

    dx, dz = dir_dict[w_degrees]

    res_point = current_point
    while True:
        # 根据朝向更新位置
        next_point_raw = (res_point[0] + dx, res_point[1] + dz)
        next_point = float_point_regularization(next_point_raw)
        # 如果下一点不在有效点中，停止
        if not check_point_in_valid_list(next_point, valid_points):
            break
        res_point = next_point

    print(f"RES Point: {res_point}")
    distance = np.linalg.norm(
        np.array(res_point) - np.array(current_point))
    return distance

# 处理并生成观察数据


def get_direction_objs(dir_index, controller, sim_args, deg_unit):

    Directions = ["Front", "Front Right", "Right",
                  "Rear Right", "Rear", "Rear Left", "Left", "Front Left"]
    Max_D = get_max_access_distance(controller, sim_args, deg_unit)
    # BaseRanges = [0, 45, 90, 135, 180, -135, -90, -45]

    event = controller.last_event
    depth_frame = event.depth_frame
    idetections = event.instance_detections2D

    scene_description = ""
    scene_objs = get_scene_objs(sim_args, idetections, depth_frame)

    # 重复检测
    uni_scene_objs = []
    for i in range(len(scene_objs)):
        each = scene_objs[i]
        if len(scene_objs) <= 1:
            scene_description += f"{each}."
            break
        if each not in uni_scene_objs:
            uni_scene_objs.append(each)
            if i >= len(scene_objs) - 1:
                scene_description += f"and {scene_objs[-1]}."
            else:
                scene_description += (each + ", ")
    if len(scene_objs) <= 0:
        scene_description = "No Interested Object."

    is_check_front = dir_index == 0
    objs_detail = get_json_objects_result(
        controller, sim_args, is_check_front_target=is_check_front)
    if objs_detail is None:
        objs_detail = "No Interested Object."

    res_data = f"""
    <Direction: {Directions[dir_index]}
    Objects detail: {objs_detail}
    Maximum distance of accessibility: "{Max_D:.2f}m"
    """
    summarize_res_data = f"<Direction: {Directions[dir_index]}>, Objects you can see: {scene_description}"
    return res_data, summarize_res_data

# 绘制当前场景可视化网格点云


def closest_grid(a, position, w):
    x, z = position
    directions = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if w not in directions:
        raise ValueError("Invalid angle. Angle should be a multiple of 45.")

    index = directions.index(w)

    if index == 0 or index == 8:  # 对应 w == 0 或 360
        closest_point = (x, z + a)
    elif index == 1:  # 对应 w == 45
        closest_point = (x + a, z + a)
    elif index == 2:  # 对应 w == 90
        closest_point = (x + a, z)
    elif index == 3:  # 对应 w == 135
        closest_point = (x + a, z - a)
    elif index == 4:  # 对应 w == 180
        closest_point = (x, z + a)
    elif index == 5:  # 对应 w == 225
        closest_point = (x - a, z - a)
    elif index == 6:  # 对应 w == 270
        closest_point = (x - a, z)
    elif index == 7:  # 对应 w == 315
        closest_point = (x - a, z + a)
    else:
        raise ValueError("Invalid angle. Angle should be in 0-360.")

    return closest_point


def find_closet_tele_pos(valid_p, obj_pos):
    min_distance = math.inf  # 初始化最小距离为无穷大
    closest_point = valid_p[0]  # 初始化最近点

    for point in valid_p:
        # 计算当前点与目标点的欧式距离
        distance = math.sqrt((point[0] - obj_pos[0])
                             ** 2 + (point[1] - obj_pos[1]) ** 2)
        # 如果当前距离小于已知的最小距离，则更新最小距离和最近点
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point


def tele_to_object_nearby(controller, tele_obj_name):
    event = controller.step(action="GetReachablePositions")

    objs = event.metadata['objects']
    reachable_pos = event.metadata['actionReturn']
    agent_pos = event.metadata['agent']['position']

    vaild_p = []
    for each in reachable_pos:
        vaild_p.append((each['x'], each['z']))

    for each in objs:
        if each["objectType"] == tele_obj_name:
            obj_pos = (each['position']['x'], each['position']['z'])
            # find a point in array valid_p that is closest to the obj_pos
            tele_pos = find_closet_tele_pos(vaild_p, obj_pos)
            event = controller.step(
                action="Teleport",
                position=dict(x=tele_pos[0], y=agent_pos['y'], z=tele_pos[1])
            )
            return event
    print(
        f"[Operation]Error: Can not find valid teleport object named {tele_obj_name}.")
    return None


def draw_current_valid_points(controller, sim_args, deg_unit):

    current_point, valid_points, w_degrees = get_grid_observation_points(
        controller, deg_unit)

    orientation_point = closest_grid(
        sim_args.GRID_SIZE, current_point, w_degrees)
    if not orientation_point:
        orientation_point = (0.0, 0.0)
    print(f"[orientation_point]: {orientation_point}")

    current_point = float_point_regularization(current_point)
    valid_points, invalid_points = get_reach_valid_points(
        sim_args.GRID_SIZE, valid_points)
    # 将点的坐标再次拆分为两个列表
    x_valid = [point[0] for point in valid_points]
    z_valid = [point[1] for point in valid_points]

    x_invalid = [point[0] for point in invalid_points]
    z_invalid = [point[1] for point in invalid_points]

    # 绘图
    # plt.scatter(x_valid, z_valid, color='red', label='Valid points')
    plt.scatter(x_valid, z_valid, color='red')
    plt.scatter(x_invalid, z_invalid, color='black', alpha=0.6)
    plt.scatter(current_point[0], current_point[1], color='green')
    plt.scatter(orientation_point[0],
                orientation_point[1], color='blue', alpha=0.8)
    plt.grid(True)
    # plt.gca().invert_yaxis()  # Z轴方向是向下的，所以需要倒转y轴
    plt.legend()
    plt.show()

#### run agent functions ####


def init_llm_bots(is_proxy=False):
    chatbot = gpt()
    chatbot.model_choose = "gpt-3.5-turbo"
    chatbot.enable_proxy = is_proxy

    summarize_bot = gpt()
    summarize_bot.model_choose = "gpt-3.5-turbo"
    summarize_bot.enable_proxy = is_proxy

    return chatbot, summarize_bot

# Functions to gengerate history.


def call_gpt_summarize_history(summarize_bot, to_summerize_history):
    gene_history_system_prompt = "You are a writing assistant tasked with writing generalizations and summaries as requested by users."
    gene_history_prompt = f"""
    Here is a single scene view from eight directions: 
    {to_summerize_history}
    Try Summarize the scene in one sentence:
    """
    success = summarize_bot.start_new_conversation(
        gene_history_system_prompt, gene_history_prompt)
    time.sleep(1)
    if success:
        reply = summarize_bot.get_last_chat_content()
    else:
        print("[ERROR]Observation History Summerizer Failed to Complete.")
        reply = "Unable to access the Observation History summary, please ignore this section."
    return reply, success


def get_history_prompt(step, summarize_bot, last_to_summerize_observation, last_thought, last_action):
    reply, success = call_gpt_summarize_history(
        summarize_bot, last_to_summerize_observation)
    history_prompt = f"""
Observation [{step}]: {reply} 
Thought [{step}]: {last_thought}
Action [{step}]: {last_action}
"""
    return history_prompt


#### prompt templates function ####

def get_full_prompt(action_type, sim_args):
    GRID_SIZE = sim_args.GRID_SIZE
    principal_prompt = f"""
You are an Embodied Agent and your task is finding a specific object in an indoor environment. You may not find it at once, so Let's work this out in a step by step way.
In the first step, you will be given target object name. During navigation, at each step, you will receive a INPUT, contains the history of the previous steps you have taken (including "Thought", "Action" and "Observation") and the observation of current viewpoint (including data of detected objects and distance you can move in 8 directions).
RULES: 
1、Your goal is to stop within 3 meters of the target object, while keeping it visiable. 
2、If the target object visible but not within 3 meters, move closer. 
3、If the target is within 3 meters but not visiable, rotate to let the object is in front of you.
"""
    action_tool_prompt = {
        OpType.DEFAULT_MOVE: f"""Use Action Tool as the following format:
- Action: [name] [parameter]

TIPS: You can only take actions from the following Actions List. You are very strict to the Actionn Name, never use nonexistent Action Names. 
Some Actions has parameters, using "[]" stands for parameter, you don't need to enter it when passing the parameter. All parameters are positive numbers with {GRID_SIZE} as the smallest unit. 

Actions List:
- MoveAhead [length] : This action means move towards your Front direction by [length] meters, default with {GRID_SIZE} meters
- RotateRight [degrees] : This action means rotate to your right by a certain angle. e.g. rotate to Front Right, degrees=45, rotate to Right, degrees=90, rotate to Rear Left, degrees=215
- Done
""",
        OpType.EIGHT_DIR_MOVE: f"""Take action to move around the room. Use Action Tool as this format:
- Action: Move [direction] [length]
- Action: RotateTo [direction]
- Action: Done

The "Move" action allows you to turn in a specified direction and then move forward a distance. It has two parameters: direction and length, where the [] just used to identify the parameter, you don't need to enter it when passing the parameter. If you only need to turn to a direction, use "RotateTo" action.

You are very strict to next two rules: 
(1) the length parameter MUST be an integer multiple of {GRID_SIZE} in meters. 
(2) The direction parameter can only be choosen from this list:
	[Front, Front Right, Right, Rear Right, Rear, Rear Left, Left, Front Left]
""",
        OpType.FOLLOW_OBJ: f"""You can use make a move to next viewpoint which is closest to an certain object. Use Action Tool as this format:
- Action: MoveTo [object_name]
- Action: RotateTo [direction]
- Action: Done

The "MoveTo" action allows you to move to the closest position to a specific object. If you only need to turn to an object, use "RotateTo" action.

You are very strict to next rules: 
(1) Only choose object_name that appears in this step’s observation. Never use nonexist or invisiable object_name. If you only need to turn to an object, use RotateTo action.
(2) The direction parameter can only be choosen from this list:
	[Front, Front Right, Right, Rear Right, Rear, Rear Left, Left, Front Left]
"""
    }

    nav_tips_prompt = f"""
At each step, you should consider: 
(1) According to the RULES, have you found the target object? If yes, you should output "Action: Done" to stop. Or you should continue.
(2) Consider where you are on the environment and where should be the next viewpoint to navigate in order to find the target object. Then use the Action tool, trying to move to that location. Show your reasoning in the Thought section. 
"""

    format_prompt = """Starting below, you should follow this format:
---
Target: (the name of the target object, always keep it in mind)
Initial Observation [1]: (initial observation of the environment)
Thought [1]: (always think about what to do next and why)
Action [1]: (the action to take, must be "Move" or "Done". If you take "Move" action, passing vaild parameters)
Observation [2]: (next observation after you took an action)
... 
(this Observation/Thought/Action pattern can repeat N times)
...
Thought [N]: I have found the target object, I can stop. 
Action [N]: Done
"""

    return principal_prompt + action_tool_prompt[action_type] + nav_tips_prompt + format_prompt


def get_observation_prompt():
    observation_tips_prompt = f"""
    TIPS: Each object in "Objects detail" section is in this format:" '(name)':'(approximate position, based on current view) (distance)' "
    And the Maximum distance of accessibility sectionis the max distance you can travel in that direction without hitting an obstacle.
"""
    return observation_tips_prompt


def get_first_prompt(target_object_type, init_observation):
    first_prompt = f"""
Begin!

Target: {target_object_type}
Initial Observation [1]: {init_observation} 
"""
    return first_prompt

#### GPT Action data function  ####


# def extract_thought(string):
#     # 先解析Thought部分，再解析Action部分
#     thought_pattern = r"Thought: (.*)\s*Action:\s*(.*)"
#     thought_match = re.match(thought_pattern, string)
#     if thought_match:
#         thought_str = thought_match.group(1)
#         action_str = thought_match.group(2)
#         return (thought_str, action_str)
#     else:
#         print("[Output Extract]The input string does not match the THOUGHT pattern.")
#         return None

def extract_action_command(action_type, action_str):
    print(f"[EXTRACT ACTION] action str = {action_str}")
    res_dict = {}
    if action_type == OpType.DEFAULT_MOVE:
        pattern = r"\s*(\w+)\s*([-+]?[0-9]*\.?[0-9]+)"
        match = re.search(pattern, action_str)

        if match:
            name = match.group(1)
            param = float(match.group(2)) if match.group(2) else None
            res_dict = {
                'name': name,
                'param': param,
                'action_type': OpType.DEFAULT_MOVE
            }
            return res_dict
        else:
            print("[Output Extract]The input string does not match the ACTION pattern.")
            return None

    if action_type == OpType.EIGHT_DIR_MOVE:
        action_pattern = r"\s*(Move)\s*(.*?)\s*([-+]?[0-9]*\.?[0-9]+)|\s*(RotateTo)\s*(.*)|^(Done)$"
        action_match = re.match(action_pattern, action_str)
        if action_match:
            if action_match.group(1):  # 匹配到Move
                action_name = action_match.group(1)
                param1 = action_match.group(2)
                param2 = float(action_match.group(3))
                res_dict = {
                    'name': action_name,
                    'direction': param1,
                    'value': param2,
                    'action_type': OpType.EIGHT_DIR_MOVE
                }
                return res_dict
            elif action_match.group(4):  # 匹配到RotateTo
                action_name = action_match.group(4)
                param1 = action_match.group(5)
                res_dict = {
                    'name': action_name,
                    'direction': param1,
                    'value': 0.0,
                    'action_type': OpType.EIGHT_DIR_MOVE
                }
                return res_dict
            elif action_match.group(6):  # 匹配到Done
                action_name = action_match.group(6)
                res_dict = {
                    'name': action_name,
                    'direction': "",
                    'value': 0.0,
                    'action_type': OpType.EIGHT_DIR_MOVE
                }
                return res_dict
        else:
            print("[Output Extract]The input string does not match the ACTION pattern.")
            return None

    if action_type == OpType.FOLLOW_OBJ:
        action_pattern = r"\s*(MoveTo|RotateTo)\s*(.*)\s*|^(Done)$"
        action_match = re.match(action_pattern, action_str)
        if action_match:
            if action_match.group(3):
                res_dict = {
                    'name': action_match.group(3),
                    'param': "",
                    'action_type': OpType.FOLLOW_OBJ
                }
            else:
                action_name = action_match.group(1)
                param1 = action_match.group(2)
                res_dict = {
                    'name': action_name,
                    'param': param1,
                    'action_type': OpType.FOLLOW_OBJ
                }
            return res_dict
        else:
            print("[Output Extract]The input string does not match the ACTION pattern.")
            return None

# do action, return event
# 解析器支持三种不同粒度格式的prompt


def do_action_by_message(controller, sim_args, action_type, msg_string: str):
    task_success = False

    # 默认参数值
    direction = 0
    distance = sim_args.GRID_SIZE

    # 只截取输出出现的第一组 Thought 和 Action
    raw_msgs = msg_string.split('\n')
    thought_str = ""
    action_str = ""
    find_flag = False
    for s in raw_msgs:
        pattern_t = r'^Thought\s*\[(\d+)\]:\s*(.*)$'
        match = re.match(pattern_t, s)
        if match:
            thought_str = match.group(2)
        else:
            pattern_a = r'^Action\s*\[(\d+)\]:\s*(.*)$'
            match = re.match(pattern_a, s)
            if match:
                action_str = match.group(2)
                find_flag = True
                break
    if not find_flag:
        print("[Operation]Error: receiving action output string with wrong format.")
        return None

    result = extract_action_command(action_type, action_str)
    if not result:
        print("[Operation]Error: receiving action output string with wrong format.")
        return None
    result_type = result['action_type']

    event = None

    if result_type == OpType.DEFAULT_MOVE:
        action_name = result['name']
        param_value = result['param']
        ACTION_MOVE_LIST = ["MoveAhead", "MoveBack"]
        ACTION_ROTATE_LIST = ["RotateLeft", "RotateRight"]
        ACTION_OTHER_LIST = ["LookUp", "LookDown"]
        if action_name not in ACTION_MOVE_LIST+ACTION_ROTATE_LIST+ACTION_OTHER_LIST:
            print("[Operation]Error: receiving action name is invaild.")
            return None
        elif not param_value:
            print("[Operation]event = controller.step(%s)" % action_name)
            event = controller.step(action_name)
        else:
            if action_name in ACTION_MOVE_LIST:
                print("[Operation]event = controller.step(%s, %f)" %
                      (action_name, param_value))
                event = controller.step(action_name, moveMagnitude=param_value)
            elif action_name in ACTION_ROTATE_LIST:
                print("[Operation]event = controller.step(%s, %f)" %
                      (action_name, param_value))
                event = controller.step(action_name, degrees=param_value)
            elif action_name == "Done":
                event = controller.step("Done")
                task_success = True
            else:
                print("[Operation]Error: receiving action name is invaild.")
                return None

    elif result_type == OpType.EIGHT_DIR_MOVE:
        action_name = result['name']
        direction_str = result['direction']
        value_float = result['value']

        if action_name == "Move":
            if direction_str in DIRECTION_DICT.keys():
                direction = DIRECTION_DICT[direction_str]
            else:
                print("[Operation]Error: receiving direction name is invaild.")
                return None
            distance = value_float
            # 执行旋转动作
            if direction > 0:
                print(
                    "[Operation]event = controller.step(RotateRight, %f)" % direction)
                event = controller.step("RotateRight", degrees=direction)
            else:
                event = controller.last_event  # 不执行动作，返回上一步的动作
            # 执行移动动作
            if distance <= 0.1 * sim_args.GRID_SIZE:
                print("[Operation]Do not move ahead, raw distance: %f)" %
                      distance)
            else:
                print(
                    "[Operation]event = controller.step(MoveAhead, %f)" % distance)
                event = controller.step(
                    "MoveAhead", moveMagnitude=distance)
        elif action_name == "RotateTo":
            if direction_str in DIRECTION_DICT.keys():
                direction = DIRECTION_DICT[direction_str]
            else:
                print("[Operation]Error: receiving direction name is invaild.")
                return None
            if direction > 0:
                print(
                    "[Operation]event = controller.step(RotateRight, %f)" % direction)
                event = controller.step("RotateRight", degrees=direction)
        elif action_name == "Done":
            event = controller.step("Done")
            task_success = True
        else:
            print("[Operation]Error: receiving action name is invaild.")
            return None

    elif result_type == OpType.FOLLOW_OBJ:
        action_name = result['name']
        param_name = result['param']

        if action_name == "MoveTo":
            tele_obj_name = param_name
            print("[Operation]Teleport to object %s " % tele_obj_name)
            event = tele_to_object_nearby(controller, tele_obj_name)
        elif action_name == "RotateTo":
            if param_name in DIRECTION_DICT.keys():
                direction = DIRECTION_DICT[param_name]
            else:
                print("[Operation]Error: receiving direction name is invaild.")
                return None
            if direction > 0:
                print(
                    "[Operation]event = controller.step(RotateRight, %f)" % direction)
                event = controller.step("RotateRight", degrees=direction)
        elif action_name == "Done":
            event = controller.step("Done")
            task_success = True
        else:
            print("[Operation]Error: receiving action name is invaild.")
            return None

    return event, thought_str, action_str, task_success


def write_chat_log(filepath, msg_list):
    with open(filepath, 'w') as f:
        for each in msg_list:
            s = each + "\n"
            f.write(s)


def write_record_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)

### ---- Class Sections ---- ###


class OpType(Enum):
    DEFAULT_MOVE = 0
    EIGHT_DIR_MOVE = 1
    FOLLOW_OBJ = 2


class SimArgs:

    def __init__(self, width, height, fov, gird_size, vis_distance, use_vaild_objs) -> None:

        # define const variables
        self.VIEW_WIDTH = width
        self.VIEW_HEIGHT = height
        self.FIELD_OF_VIEW = fov
        self.GRID_SIZE = gird_size
        self.VISIBLE_DISTANCE = vis_distance
        self.USE_VAILD_OBJ_TYPES = use_vaild_objs
        self.rules_achieve = False  # 模拟器检验任务是否完成
        self.vaild_obj_types = []
        self.target_object_type = ""


class Agent:

    def __init__(self, action_type=OpType.EIGHT_DIR_MOVE, grid_size=0.25) -> None:

        self.SIM_ARGS = SimArgs(640, 480, 45, grid_size, 3, True)
        self.DEG_UNIT = 45  # 每次旋转的间隔角度
        self.MAX_STEP = 16
        self.ACTION_TYPE = action_type
        # 初始化llm bot服务
        self.chatbot, self.summarize_bot = init_llm_bots()

    # variables
    controller = None
    step_count = 0
    task_success = False  # LLM输出的任务是否完成

    init_observation = ""
    init_summarize = ""

    chat_logs = []  # 记录日志
    history_result_prompts = []
    paths = []  # 记录移动轨迹

    log_file_path = "./gpt_output.txt"
    record_file_path = "./test_path_record.txt"

    ### debug section ###
    def draw_position_points(self):
        if self.controller:
            draw_current_valid_points(
                self.controller, self.SIM_ARGS, self.DEG_UNIT)

    def make_controller_teleport_to_object(self, object_name):
        tele_to_object_nearby(self.controller, tele_obj_name=object_name)

    def make_controller_single_action(self, act_name, act_param=None):
        ACTION_MOVE_LIST = ["MoveAhead", "MoveBack"]
        ACTION_ROTATE_LIST = ["RotateLeft", "RotateRight"]
        ACTION_OTHER_LIST = ["LookUp", "LookDown", "Done"]
        if act_name not in ACTION_MOVE_LIST+ACTION_ROTATE_LIST+ACTION_OTHER_LIST:
            print("[Operation]Error: receiving action name is invaild.")
            return None
        elif not act_param:
            print("[Operation]event = controller.step(%s)" % act_name)
            event = self.controller.step(act_name)
        else:
            if act_name in ACTION_MOVE_LIST:
                print("[Operation]event = controller.step(%s, %f)" %
                      (act_name, act_param))
                event = self.controller.step(act_name, moveMagnitude=act_param)
            elif act_name in ACTION_ROTATE_LIST:
                print("[Operation]event = controller.step(%s, %f)" %
                      (act_name, act_param))
                event = self.controller.step(act_name, degrees=act_param)

        if (event.metadata['lastActionSuccess']):
            print(event.metadata['agent'])
        else:
            print("No vaild move! Error: %s" % event.metadata["errorMessage"])
        self.controller.step("Done")
        return event

    def make_controller_do_move(self, msg_string):
        msg_string = msg_string.replace('\n', '')
        result = extract_action_command(OpType.EIGHT_DIR_MOVE, msg_string)
        if not result:
            print("[Operation]Error: receiving action output string with wrong format.")
            return None

        action_str, direction_str, value_float = result

        if action_str == "Move":
            if direction_str in DIRECTION_DICT.keys():
                direction = DIRECTION_DICT[direction_str]
            else:
                print("[Operation]Error: receiving direction name is invaild.")
                return None
            distance = value_float
            # 执行旋转动作
            if direction > 0:
                print(
                    "[Operation]event = controller.step(RotateRight, %f)" % direction)
                event = self.controller.step("RotateRight", degrees=direction)
            # 执行移动动作
            print("[Operation]event = controller.step(MoveAhead, %f)" % distance)
            event = self.controller.step("MoveAhead", moveMagnitude=distance)
        self.controller.step("Done")
        return event
    ### debug section ###

    def write_records(self):
        write_chat_log(self.log_file_path, self.chat_logs)
        path_record = json.dumps(self.paths)
        write_record_file(self.record_file_path, path_record)

    def get_first_history_prompt(self, last_thought, last_action):
        reply, success = call_gpt_summarize_history(
            self.summarize_bot, self.init_summarize)
        first_history_prompt = f"""
Begin!

Target: {self.SIM_ARGS.target_object_type}
Initial Observation [1]: {reply} 
Thought [1]: {last_thought}
Action [1]: {last_action}"""
        return first_history_prompt

    def get_init_step_input(self):
        obs, s_obs = get_observation(
            self.controller, self.SIM_ARGS, self.DEG_UNIT)
        res = ""
        for each in obs:
            res += each
        res += get_observation_prompt()

        # return: observation, summrize_oberservation
        return res, "\n".join(s_obs)

    def get_next_step_input(self, step, thought, action, history_list):
        obs, summarize_obs = get_observation(
            self.controller, self.SIM_ARGS, self.DEG_UNIT)
        obs_str = ""
        for each in obs:
            obs_str += each
        obs_str += get_observation_prompt()

        if step == 1:
            new_history_str = self.get_first_history_prompt(thought, action)
        else:
            new_history_str = get_history_prompt(
                step, self.summarize_bot, summarize_obs, thought, action)

        # 加入总结历史
        history_list.append(new_history_str)

        # 构造此步骤的输出
        next_input_str = ""
        for each in history_list:
            next_input_str += each

        next_input_str += f"""Observation [{step + 1}]: \n {obs_str}"""
        return next_input_str

    def init_epsoide(self, ep):
        self.step_count = 0  # 重置step count
        self.chat_logs.clear()
        self.history_result_prompts.clear()
        self.paths.clear()
        self.SIM_ARGS.rules_achieve = False

        # 先调用，以获得Controller和初始位置event
        self.controller, init_event, self.SIM_ARGS.target_object_type = start_scene_by_episode(
            self.SIM_ARGS, ep)
        # 获取有效物体列表
        self.SIM_ARGS.vaild_obj_types = get_vaild_obj_types(init_event)
        # 获得初始观察
        self.init_observation, self.init_summarize = self.get_init_step_input()

    def reset_epsoide(self, ep):
        self.step_count = 0  # 重置step count
        self.chat_logs.clear()
        self.history_result_prompts.clear()
        self.paths.clear()
        self.SIM_ARGS.rules_achieve = False

        init_event, self.SIM_ARGS.target_object_type = reset_scene_by_episode(
            self.controller, self.SIM_ARGS, ep)
        # 获取有效物体列表
        self.SIM_ARGS.vaild_obj_types = get_vaild_obj_types(init_event)
        # 获得初始观察
        self.init_observation, self.init_summarize = self.get_init_step_input()

    def mannual_do_first_step(self, ep):
        # 初始化模拟器
        self.init_epsoide(ep)

        gpt_prompt = get_full_prompt(self.ACTION_TYPE,
                                     self.SIM_ARGS) + get_first_prompt(self.SIM_ARGS.target_object_type, self.init_observation)

        if gpt_prompt:
            with open("./mannual_step_to_write.txt", 'w') as f:
                f.write(gpt_prompt)
            return
    # 手动执行使用，通过读写文件操作

    def mannual_do_step(self):
        MAX_STEP = self.MAX_STEP
        try:
            self.step_count += 1
            if self.step_count > MAX_STEP:
                print("Error: MAX step tries reached.")
                return False

            reply = ""
            with open("./mannual_step_to_read.txt", 'r') as f:
                reply = f.read()

            print(f"<STEP:{self.step_count}reply: {reply}")

            # 先写日志，防止抛出异常
            self.chat_logs.append(f"<STEP:{self.step_count}\n ")
            self.chat_logs.append(reply)

            do_action_return = do_action_by_message(
                self.controller,
                self.SIM_ARGS,
                action_type=self.ACTION_TYPE,
                msg_string=reply)

            if do_action_return is None:
                print("Error: do action failed.")
                return False

            else:
                event, thought, action, task_success = do_action_return
                next_observation_input = self.get_next_step_input(
                    step=self.step_count,
                    thought=thought,
                    action=action,
                    history_list=self.history_result_prompts
                )
                # 写入路径记录
                self.paths.append({
                    "point": event.metadata['agent']['position'],
                    "rules_achieve": self.SIM_ARGS.rules_achieve
                })

            # print(
            #     f"<STEP:{self.step_count}history_result_prompts len: {len(self.history_result_prompts)}")

            if not event.metadata['lastActionSuccess']:
                print("Error: Agent did invaild move! Detail: %s" %
                      event.metadata["errorMessage"])
                return False

            if task_success:
                print("<Task Success Done!>")
                self.task_success = True
            else:
                print("<This Step Did Not Finish The Task.>")

            self.write_records()

            next_gpt_prompt = get_full_prompt(self.ACTION_TYPE,
                                              self.SIM_ARGS) + next_observation_input
            with open("./mannual_step_to_write.txt", 'w') as f:
                f.write(next_gpt_prompt)

            return True  # 标志步骤执行成功
        except Exception as e:
            print(f"<Task exceeded by exception! Detail: {e}>")
            self.write_records()

            return False

    # 全自动流程
    # 首次调用时，请先执行 init ep 以获得Controller
    def auto_do_nav_episode(self, ep):
        MAX_STEP = self.MAX_STEP
        task_success = False

        gpt_prompt = ""
        next_observation_input = ""

        # 初始化模拟器
        self.reset_epsoide(ep)

        sys_prompt = "You are an Embodied Agent, who interact with the environment and complete tasks for human."  # 初始化 system prompt

        try:
            while not task_success:

                self.step_count += 1

                if self.step_count > MAX_STEP:
                    print("Error: MAX step tries reached.")
                    break

                if self.step_count == 1:
                    gpt_prompt = get_full_prompt(self.ACTION_TYPE,
                                                 self.SIM_ARGS) + get_first_prompt(self.SIM_ARGS.target_object_type, self.init_observation)
                else:
                    gpt_prompt = get_full_prompt(self.ACTION_TYPE,
                                                 self.SIM_ARGS) + next_observation_input
                # print(f"<STEP:{step_count}>, prompt: {gpt_prompt}")

                success = self.chatbot.start_new_conversation(
                    sys_prompt, gpt_prompt)
                time.sleep(1)
                if not success:
                    print("Error: call chatbot failed.")
                    break
                reply = self.chatbot.get_last_chat_content()
                print(f"<STEP:{self.step_count}reply: {reply}")

                # 先写日志，防止抛出异常
                self.chat_logs.append(f"<STEP:{self.step_count}\n ")
                self.chat_logs.extend(self.chatbot.get_messages_list())

                do_action_return = do_action_by_message(
                    self.controller,
                    self.SIM_ARGS,
                    action_type=self.ACTION_TYPE,
                    msg_string=reply)

                if do_action_return is None:
                    print("Error: do action failed.")
                    break
                else:
                    event, thought, action, task_success = do_action_return
                    next_observation_input = self.get_next_step_input(
                        step=self.step_count,
                        thought=thought,
                        action=action,
                        history_list=self.history_result_prompts
                    )
                    # 写入路径记录
                    self.paths.append({
                        "point": event.metadata['agent']['position'],
                        "rules_achieve": self.SIM_ARGS.rules_achieve
                    })

                    if not event.metadata['lastActionSuccess']:
                        print("Error: Agent did invaild move! Detail: %s" %
                              event.metadata["errorMessage"])
                        break
                # print(
                #     f"<STEP:{self.step_count}history_result_prompts len: {len(self.history_result_prompts)}")
            self.write_records()

            if task_success or self.SIM_ARGS.rules_achieve:
                print("<Task Success Done!>")
            else:
                print("<Task Finished. But end up failed.>")

            return (task_success and 1 or 0, self.SIM_ARGS.rules_achieve and 1 or 0, True)

        except Exception as e:
            print(f"<Task exceeded by exception! Detail: {e}>")
            self.write_records()
            return (0, 0, False)
