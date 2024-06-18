from zhipuai import ZhipuAI
import json
import requests
from http import HTTPStatus
import dashscope
import re
from openai import OpenAI
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage


# 智谱AI大模型：https://open.bigmodel.cn/
chatglm_api=""

# 阿里云千问系列大模型（请确保所有模型开通计费管理！！！记得充钱）：https://dashscope.console.aliyun.com/overview
qwen_api=''

# 百度千帆大模型（请确保所有模型开通计费管理！！！记得充钱）：https://console.bce.baidu.com/qianfan/overview
qianfan_api={
    "api_key":"",
    "secret_key":""
}

# deepseek：https://platform.deepseek.com/usage
deepseek_api=""

# 星火大模型：https://console.xfyun.cn/services/cbm
sparkai_api={
    "SPARKAI_APP_ID" : '',
    "SPARKAI_API_SECRET" : '',
    "SPARKAI_API_KEY" : '',
}

# 暗之月面：https://platform.moonshot.cn/console/info
kimi_api=""
# 小小的备注
"""
目前使用来看qwen-max>=ERNIE-4.0系列>kimi>=qwen1.5-110b-chat>=deepseek>=glm4>sparkai-3.5
但是kimi送了好多api，不用浪费了！
deepseek送的也多
星火大模型虽然也多，但是由于它输入的格式是没有上下文的！并不太推荐使用这个，当然对于一问一答的一些常规的问题还是能用的
"""

model_qianfan_dir={"ERNIE-3.5-8K":"completions",
                   "ERNIE-Speed-8K":"ernie_speed",
                   "ERNIE-3.5-8K-0205":"ernie-3.5-8k-0205",
                   "ERNIE-3.5-8K-1222":"ernie-3.5-8k-1222",
                   "ERNIE-3.5-4K-0205":"ernie-3.5-4k-0205",
                   "ERNIE-3.5-8K（抢占式）":"completions_preemptible",
                   "ERNIE-3.5-8K-Preview":"ernie-3.5-8k-preview",
                   "ERNIE-Speed-128K(预览版)":"ernie-speed-128k",
                   "ERNIE-Lite-8K-0922":"eb-instant",
                   "ERNIE-Lite-8K-0308":"ernie-lite-8k",
                   "ERNIE-Lite-128K-0419":"调用api",
                   "ERNIE-Tiny-8K":"ernie-tiny-8k",
                   "ERNIE-Character-8K-0321":"ernie-char-8k",
                   "ERNIE-Functions-8K-0321":"YOU API",
                   "Gemma-7B-it":"gemma_7b_it",
                   "Yi-34B-Chat":"yi_34b_chat",
                   "Mixtral-8x7B-Instruct":"mixtral_8x7b_instruct",
                   "Mistral-7B-Instruct":"YOU API",
                   "Qianfan-Chinese-Llama-2-7B-32K":"YOU API",
                   "Qianfan-Chinese-Llama-2-7B":"qianfan_chinese_llama_2_7b",
                   "llama2-13b-chat":"llama_2_13b",
                   "llama3-8b-instruct":"llama_3_8b",
                   "ERNIE-4.0-8K-0329":"ernie-4.0-8k-0329",
                   "ERNIE-4.0-8K-0104":"ernie-4.0-8k-0104",
                   "ERNIE-4.0-8K":"completions_pro",
                   "ERNIE-3.5-128K":"ernie-3.5-128k"}
class llm_model():
    def __init__(self):
        # 不再使用这个了
        self.model_list={
            # 智谱ai系列
            "glm-4":self.chatglm_model,
            "glm-3-turbo":self.chatglm_model,
            # 千问大模型系列
            "qwen-turbo": self.qwen_model,
            "qwen-plus": self.qwen_model,
            "qwen-max":self.qwen_model,
            "qwen-max-0428": self.qwen_model,
            "qwen-max-0403": self.qwen_model,
            "qwen-max-0107": self.qwen_model,
            "qwen-7b-chat": self.qwen_model,
            "qwen-14b-chat": self.qwen_model,
            "qwen-72b-chat": self.qwen_model,
            "qwen1.5-7b-chat": self.qwen_model,
            "qwen1.5-14b-chat": self.qwen_model,
            "qwen1.5-72b-chat": self.qwen_model,
            "qwen1.5-110b-chat": self.qwen_model,
            # 千帆大模型系列
            "ERNIE-3.5-8K":self.qianfan_model,
            "ERNIE-3.5-8K-0205":self.qianfan_model,
            "ERNIE-4.0-8K-0104": self.qianfan_model,
            "ERNIE-4.0-8K-0329": self.qianfan_model,
            # 暗之月面模型
            "deepseek-chat":self.deepseek_model,
            # 星火大模型
            "星火大模型v3.5": self.sparkai_model,
            "星火大模型v3.0": self.sparkai_model,
            "星火大模型v2.0": self.sparkai_model,
            "星火大模型v1.5": self.sparkai_model,
            # 暗之月面大模型kimi
            "moonshot-v1-8k":self.kimi_model,
            "moonshot-v1-32k": self.kimi_model,
            "moonshot-v1-128k": self.kimi_model,
        }

        self.model_qwen_list=["qwen-turbo","qwen-long","qwen-plus","qwen-max","qwen-max-0428","qwen-max-0403","qwen-max-0107","qwen-7b-chat","qwen-14b-chat","qwen-72b-chat","qwen1.5-7b-chat","qwen1.5-14b-chat","qwen1.5-72b-chat","qwen1.5-110b-chat"]
        self.model_qianfan_list=["ERNIE-3.5-8K","ERNIE-3.5-128K","ERNIE-3.5-8K-0205","ERNIE-4.0-8K","ERNIE-4.0-8K-0104","ERNIE-4.0-8K-0329","ERNIE-3.5-8K-0205","ERNIE-Speed-128K(预览版)"]
        self.model_zhipuai_list=["glm-4","glm-3-turbo","glm-4-0520","glm-4-air","glm-4-airx","glm-4-flash"]
        self.model_kimi_list=["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"]
        self.model_sparkai_list=["星火大模型v3.5","星火大模型v3.0"]
        self.model_deepseek_list=["deepseek-chat"]

    # 熟悉的智谱ai
    def chatglm_model(self,model,question):
        client = ZhipuAI(api_key=chatglm_api)
        response = client.chat.completions.create(
            model=model,  # 填写需要调用的模型名称
            messages=question,
        )
        return response.choices[0].message.content

    # 千问大模型
    def qwen_model(self,model,question):
        dashscope.api_key = qwen_api
        response = dashscope.Generation.call(model=model,
                                   messages=question,
                                   # 将输出设置为"message"格式
                                   result_format='message')

        # 检查响应是否成功
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content']
        else:
            print("请求失败:", response.status_code,"返回错误信息：",response)

    # 百度千帆大模型
    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}".format(
            qianfan_api["api_key"], qianfan_api["secret_key"])
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def qianfan_model(self,model,question):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{}?access_token=".format(
            model_qianfan_dir[model]) + self.get_access_token()
        # print(url)
        payload = json.dumps({
            "messages": question
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        text = re.findall(r'"result":(.*?),', response.text)[0].replace("\n", "")
        return text

    # deepseek大模型(这里的model是为了统一格式，也为了后续推出更多的model)
    def deepseek_model(self,model,question):
        client = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=question,
            stream=False
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content

    # 星火大模型
    def sparkai_model(self,model,question):
        if model == "星火大模型v3.5":
            SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
            SPARKAI_DOMAIN = 'generalv3.5'
        elif model == "星火大模型v3.0":
            SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.1/chat'
            SPARKAI_DOMAIN = 'generalv3'
        # 2.0已经被删除了
        elif model == "星火大模型v2.0":
            SPARKAI_URL = 'wss://spark-api.xf-yun.com/v2.1/chat'
            SPARKAI_DOMAIN = 'generalv2'
        elif model == "星火大模型v1.5":
            SPARKAI_URL = 'wss://spark-api.xf-yun.com/v1.1/chat'
            SPARKAI_DOMAIN = 'general'
        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=sparkai_api["SPARKAI_APP_ID"],
            spark_api_key=sparkai_api["SPARKAI_API_KEY"],
            spark_api_secret=sparkai_api["SPARKAI_API_SECRET"],
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
        messages = [ChatMessage(
            role="user",
            content=question
        )]
        handler = ChunkPrintHandler()
        a = spark.generate([messages], callbacks=[handler])
        b = list(a)
        response=b[0][1][0][0].text
        return response
    # kimi大模型
    def kimi_model(self,model,question):
        client = OpenAI(
            api_key=kimi_api,
            base_url="https://api.moonshot.cn/v1",
        )

        completion = client.chat.completions.create(
            model=model,
            messages=question,
            temperature=0.3,
        )

        # print(completion.choices[0].message.content)
        return completion.choices[0].message.content
