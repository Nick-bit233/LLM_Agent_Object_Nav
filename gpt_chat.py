import requests
import time
import json


def load_dict_from_file(config_fp):
    with open(config_fp, 'r') as file:
        dicts = json.load(file)
    return dicts


class gpt:
    def __init__(self, platform="openai", proxy=False, max_token=4000) -> None:
        self.platform_name = platform
        self.enable_proxy = proxy
        self.MAX_TOKEN_LENGTH = max_token

        config_fp = "./llm_chat_config.json"
        dicts = load_dict_from_file(config_fp)
        self.URLS = dicts["URLS"]
        self.API_KEYS = dicts["API_KEYS"]
        self.proxies = dicts["proxies"]

    CHAT_SUFFIX = "v1/chat/completions"
    MODEL_SUFFIX = "v1/models"
    model_choose = "gpt-3.5-turbo"  # 默认调用的模型名称

    # 存储一轮对话中的消息
    messages = []
    total_usage = 0
    last_chat_success = True

    def send_request_model_list(self):
        # 请求头
        headers = {
            "Authorization": f"Bearer {self.API_KEYS[self.platform_name]}"
        }
        # 发送请求
        if self.enable_proxy:
            response = requests.get(
                self.URLS[self.platform_name] + self.MODEL_SUFFIX, headers=headers, proxies=self.proxies)
        else:
            response = requests.get(
                self.URLS[self.platform_name] + self.MODEL_SUFFIX, headers=headers)

        # 解析响应
        if response.status_code == 200:
            data = response.json()
            model_list = data["data"]
            print(model_list)
            return model_list
        else:
            print("response code: %d, error msg: %s" %
                  (response.status_code, response.text))
            return None

    def send_request_chat(self, messages):
        # 请求参数
        parameters = {
            "model": self.model_choose,
            # [{"role": "user", "content": context}]
            "messages": messages
        }
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEYS[self.platform_name]}"
        }
        # 发送请求
        max_timeout = 120
        # add retries
        max_retries = 3
        for i in range(max_retries):
            try:
                if self.enable_proxy:
                    response = requests.post(
                        self.URLS[self.platform_name] + self.CHAT_SUFFIX, headers=headers, json=parameters, proxies=self.proxies, timeout=max_timeout)
                else:
                    response = requests.post(
                        self.URLS[self.platform_name] + self.CHAT_SUFFIX, headers=headers, json=parameters, timeout=max_timeout)
                # 解析响应
                if response.status_code == 200:
                    data = response.json()
                    usage = data["usage"]["total_tokens"]
                    msg = data["choices"][0]["message"]
                    self.total_usage = usage

                    return msg
                else:
                    print("No vaild response! Response code: %d, error msg: %s" %
                          (response.status_code, response.text))
                    return None

            except requests.Timeout:
                print(f"Request timed out! Retry {i+1}/{max_retries}")
            except requests.RequestException as e:
                print(f"Request error: {e} Retry {i+1}/{max_retries}")

        return None

    def start_new_conversation(self, system_input, user_input):
        new_messages = [{"role": "system", "content": system_input}]
        self.messages = new_messages
        self.total_usage = 0
        return self.continue_conversation(user_input)

    def continue_conversation(self, user_input):
        if self.total_usage > self.MAX_TOKEN_LENGTH:
            self.last_chat_success = False
            print("Waring: The chat has over the token limit, Please start a new chat")
            return False

        user_message = {"role": "user", "content": user_input}

        # 将用户输入添加到messages中
        self.messages.append(user_message)

        # 发送API请求
        response_msg = self.send_request_chat(self.messages)

        # 输出API返回内容
        if response_msg:
            self.last_chat_success = True
            # print("[GPT3.5]: ")
            # print(response_text["content"])
            # 将API接口返回的内容添加至messages，以用作多轮对话
            self.messages.append(response_msg)
            return True
        else:
            self.last_chat_success = False
            print("ERROR: This Chat Failed, No message will be added. Please Try Again.")
            return False

    def get_last_chat_content(self):
        if self.last_chat_success:
            msg = self.messages[-1]
            if msg["role"] == "assistant":
                return msg["content"]
            else:
                return "[This is an User input]: " + msg["content"]
        else:
            return "Last Chat Failed."

    def get_messages_list(self):
        res = []
        for each in self.messages:
            if each["role"] == "user":
                res.append("User:" + each["content"])
            elif each["role"] == "assistant":
                res.append("ChatGPT:" + each["content"])
            elif each["role"] == "system":
                res.append("[System Prompt]:" + each["content"])
        res.append("Total_tokens: %d" % self.total_usage)
        return res


# test call llm
if __name__ == '__main__':
    bot = gpt(platform="closeai", proxy=False)
    bot.model_choose = "gpt-3.5-turbo"

    # mlist = bot.send_request_model_list()
    # for i in mlist:
    #     print(i["id"])

    sys_prompt = "You are an Embodied Agent assistant."  # 初始化 system prompt
    user_prompt = "Hello!"  # 初始化 user prompt
    bot.start_new_conversation(sys_prompt, user_prompt)
    reply = bot.get_last_chat_content()
    print(f"reply: {reply}")
