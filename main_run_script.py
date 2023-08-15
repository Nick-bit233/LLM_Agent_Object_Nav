import time
import os
import json
import sys
import glob
import gzip
from llm_obj_nav_agent import Agent, OpType
import errno

# load dataset function


def load_split(dataset_dir, split):
    # dataset_dir: root path of dataset
    # split: either "debug","train","val" or "test"
    # each dataset contains many episodes, which are all ".json.gz" zip file.
    split_paths = os.path.join(dataset_dir, split, "episodes", "*.json.gz")
    split_paths = sorted(glob.glob(split_paths))

    episode_list = []
    dataset = {}

    for split_path in split_paths:
        # print("Loading: {path}".format(path=split_path))

        with gzip.GzipFile(split_path, "r") as f:
            episodes = json.loads(f.read().decode("utf-8"))

            # Build a dictionary of the dataset indexed by scene, object_type
            curr_scene = None
            curr_object = None
            points = []
            scene_points = {}
            for data_point in episodes:
                if curr_object != data_point["object_type"]:
                    scene_points[curr_object] = points
                    curr_object = data_point["object_type"]
                    points = []
                if curr_scene != data_point["scene"]:
                    dataset[curr_scene] = scene_points
                    curr_scene = data_point["scene"]
                    scene_points = {}

                points.append(data_point)

            episode_list += episodes
    # reutrn type:
    #    episode_list: list of init scene and target data, for setting and val.
    #    dataset: dict of train path to nav object, for training.
    return episode_list, dataset


def load_dataset_episode(dataset_path, split_name, index):
    test_episodes, dataset = load_split(
        dataset_dir=dataset_path, split=split_name)
    # print("episodes len: %d" % len(test_episodes))
    ep = test_episodes[index]

    return ep


def make_dir_by_filepath(filepath):
    try:
        os.makedirs(os.path.dirname(filepath))
    except OSError as exc:  # 如果在创建目录时发生错误
        if exc.errno != errno.EEXIST:  # 如果错误不是“目录已经存在”的错误
            raise


#####


def get_last_processed_result(filepath):
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        return existing_data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def set_last_processed_result(filepath, res):
    with open(filepath, 'w') as f:
        f.write(json.dumps(res))


def main():
    ##### Change this exp parameters #####

    ACTION_TYPE = OpType.FOLLOW_OBJ
    GRID_SIZE = 0.1

    ######################################

    dataset_path = "/home/nickbit/ai2thor/dataset/"
    split_name = "val"  # for some reason, "test" set are not provided by aithor 5.0.0

    agent = Agent(action_type=ACTION_TYPE,
                  grid_size=GRID_SIZE)

    test_episodes, dataset = load_split(
        dataset_dir=dataset_path, split=split_name)
    print("episodes len: %d" % len(test_episodes))

    test_name = "sub_ep_test_0814_2"
    rec_rootpath = f"./test_records/{test_name}/"
    temp_result_path = f"./exp_temp_result/{test_name}.txt"

    # 读取历史数据文件（续断点）
    INIT_START_INDEX = 0
    end_index = 1800

    processed_res = get_last_processed_result(temp_result_path)
    if processed_res:
        start_index = processed_res["Finished ep index"] + 1
        true_positive_success = processed_res["SR"]
        gpt_success = processed_res["SR_GPT"]
        rule_success = processed_res["SR_RULE"]
        exceptioon_occr = processed_res["Exception cnt"]
    else:
        processed_res = {
            "Init start index": INIT_START_INDEX,
            "Finished ep index": INIT_START_INDEX,
            "SR": 0,
            "SR_GPT": 0,
            "SR_RULE": 0,
            "Exception cnt": 0
        }
        start_index = INIT_START_INDEX
        true_positive_success = 0
        gpt_success = 0
        rule_success = 0
        exceptioon_occr = 0

    length = end_index - start_index
    print(rec_rootpath)

    agent.init_epsoide(test_episodes[start_index])

    for i in range(start_index, end_index):
        ep = test_episodes[i]
        # 修改日志文件目录
        try:
            rec_path = os.path.join(rec_rootpath, f"{i}_{ep['id']}")
            agent.log_file_path = os.path.join(rec_path, "gpt_chat_log.txt")
            agent.record_file_path = os.path.join(
                rec_path, "trajectory_record_log.json")
            if not os.path.exists(os.path.dirname(agent.log_file_path)):
                make_dir_by_filepath(agent.log_file_path)
        except Exception as e:
            print(
                f"<Set log file path failed! Using default filepath. Detail: {e}>")
            agent.log_file_path = os.path.join(
                rec_rootpath, "defalut_gpt_chat_log.txt")
            agent.record_file_path = os.path.join(
                rec_rootpath, "defalut_trajectory_record_log.json")
            break

        # TODO：检查驱动BUG，并尝试自动化
        if not agent.check_session_alive():
            print(f"<图形驱动失效 终止运行 已进行的ep数量: {i - start_index}>")
            sys.exit(1)  # 非正常退出

        # 自动执行
        result = agent.auto_do_nav_episode(ep)
        if result[2]:
            gpt_success += result[0]
            rule_success += result[1]
            if result[0] == 1 and result[1] == 1:
                true_positive_success += 1
        else:
            exceptioon_occr += 1

        processed_res["Finished ep index"] = i
        processed_res["SR"] = true_positive_success
        processed_res["SR_GPT"] = gpt_success
        processed_res["SR_RULE"] = rule_success
        processed_res["Exception cnt"] = exceptioon_occr

        set_last_processed_result(temp_result_path, processed_res)  # 写入下一文件地址

    print(f"------------------RESULT------------------")
    print(f"SR = {true_positive_success / length}")
    print(f"SR_GPT = {gpt_success / length}")
    print(f"SR_RULE = {rule_success / length}")
    print(f"exception cnt = {exceptioon_occr}")


if __name__ == "__main__":
    main()
