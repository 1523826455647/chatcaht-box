import csv
import json
import re
from datetime import datetime
from llm_list import llm_model
import time
import concurrent.futures
import gradio as gr


# 认知大模型
class model_chat(llm_model):
    def __init__(self):
        super().__init__()
        self.history=[]
        self.sparkai_prompt=""
        self.history_dic={}
        self.chat_history=[]
    def clear_history(self,model):
        self.history=[]
        print(f"由于更换模型：{model}，清除历史记录！")

    def clear_chat_history(self):
        self.history=[]
        print("已经清除历史记录啦！")
        return ""

    def chat_with_llm(self,model,question,prompt,chat_history):
        if question == "":
            return "", self.chat_history
        self.chat_history=chat_history
        start_time=time.time()
        print(f"当前回答的模型为{model}")
        print(f"当前提交的问题是：{question}")
        if model in self.model_qwen_list:
            model_function=self.qwen_model
        elif model in self.model_qianfan_list:
            model_function=self.qianfan_model
        elif model in self.model_zhipuai_list:
            model_function = self.chatglm_model
        elif model in self.model_kimi_list:
            model_function=self.kimi_model
        elif model in self.model_sparkai_list:
            model_function=self.sparkai_model
        elif model in self.model_deepseek_list:
            model_function=self.deepseek_model
        # model_function=self.model_list.get(model)
        # 创建历史
        if prompt != "":
            self.history.append({"role":"system","content":prompt})
        self.history.append({"role":"user","content":question})
        if model in self.model_sparkai_list:
            # self.sparkai_prompt=self.sparkai_prompt+prompt+question
            # response=model_function(model,self.sparkai_prompt)
            # self.sparkai_prompt=self.sparkai_prompt+"/n/n/n"+response
            response = model_function(model, question)
        else:
            response=model_function(model,self.history)
            self.history.append({"role": "assistant", "content": response})
        print(f"模型{model}回答的结果为{response}")
        # 将内容加入到里面
        self.chat_history.append((question, response))
        # 这里没有星火的，他的玩法太不正宗了
        print(f"历史：{self.history}")
        end_time=time.time()
        run_time = end_time - start_time
        run_time_dt = datetime.utcfromtimestamp(run_time)
        formatted_time = run_time_dt.strftime('%H:%M:%S')
        print(f"本次回答所消耗的时间为{formatted_time}")
        return "",self.chat_history

    def save_history(self,theme):
        print(type(theme))
        with open("chat_history.json", "r", encoding="utf-8-sig") as file:
            try:
                data = json.load(file)
                for idx_data in data:
                    if theme in idx_data:
                        data.remove(idx_data)
            except:
                data=[]

        # 创建新的历史
        with open ("chat_history.json", "w", encoding="utf-8-sig") as file:
            if theme =="":
                theme=self.history[0]["content"][0:5]
            add_data=[self.history,self.chat_history]
            new_data={theme:add_data}
            data.append(new_data)
            json.dump(data, file, ensure_ascii=False)
            print("已经保存历史记录啦！")
            return theme

    def load_history(self,theme):
        with open("chat_history.json", "r", encoding="utf-8-sig") as file:
            data=json.load(file)
            # 第一个历史是给llm的，第二个是给llm-box的
            for old_data in data:
                if theme in old_data:
                    self.history=old_data[theme][0]
                    self.chat_history=old_data[theme][1]
                    print(f"切换历史聊天记录，当前历史为：{self.history}")
                    break

    def renew_history_chat(self,theme):
        # 清除原本的历史
        self.clear_chat_history()
        # 加载历史
        self.load_history(theme)
        return self.chat_history,theme

    def del_history_from_file(self,theme):
        with open("chat_history.json", "r", encoding="utf-8-sig") as file:
            try:
                data = json.load(file)
            except:
                data=[]

        with open("chat_history.json", "w", encoding="utf-8-sig") as file:
            new_data=[]
            for chat_data in data:
                if theme == list(chat_data.keys())[0]:
                    continue
                new_data.append(chat_data)
            json.dump(new_data, file, ensure_ascii=False)

    def del_history(self,theme):
        # 清除原本的历史
        self.clear_chat_history()
        # 删除历史
        self.del_history_from_file(theme)
        print(f"已经删除历史记录：{theme}！")
        return ""


class writer_expert(llm_model):
    def __init__(self):
        super().__init__()
        self.history=[]
        self.content=""

    def clear_history(self):
        self.history=[]
        self.content=""

    def clear_all(self):
        self.history=[]
        self.content=""
        return "","",""
    def llm_struct(self,model,title,type):
        struct_content = []
        self.content=self.content+title+"\n"
        print(f"当前回答的模型为{model}")
        if model in self.model_qwen_list:
            self.model_function = self.qwen_model
        elif model in self.model_qianfan_list:
            self.model_function = self.qianfan_model
        elif model in self.model_zhipuai_list:
            self.model_function = self.chatglm_model
        elif model in self.model_kimi_list:
            self.model_function = self.kimi_model
        elif model in self.model_sparkai_list:
            self.model_function = self.sparkai_model
        elif model in self.model_deepseek_list:
            self.model_function = self.deepseek_model
        struct_content.append({"role":"system","content":"你是一个文章大纲写作助手，接下来我会给你一个标题，你需要根据这个标题，回答整一个文章的写作大纲，要求文章结构完整，思路清晰"})
        self.title_prompt=f"我的文章标题为{title}，请帮我写出文章的大纲"
        struct_content.append({"role":"user","content":self.title_prompt})
        self.response=self.model_function(model,struct_content)

        writer_prompt = f"你是一个文章写作专家，你将帮助我完成一篇文章的写作，我需要的写作风格为{type}的文章，请按照这种风格帮助我完成文章的写作，你需要尽可能地多些内容，可以适当扩展一下相关内容，语言要通顺且自然，并适当举出相应的案例辅助写作。"
        self.history.append({"role": "system", "content": writer_prompt})
        self.history.append({"role": "user", "content": self.title_prompt})
        self.history.append({"role": "assistant", "content": self.response})
        print(self.history)
        return self.response

    def generate_content(self,model,section):
        section_prompr=f"开始写作章节为：{section}"
        self.history.append({"role":"user","content":section_prompr})
        self.response_content=self.model_function(model,self.history)
        self.history.append({"role":"assistant","content":self.response_content})
        # print("重新之前",self.history)
        return self.response_content

    def rewriter_content(self,model,section):
        self.history=self.history[0:-2]
        self.response_content=self.generate_content(model,section)
        # print("重新之后",self.history)
        return self.response_content

    def sumbit_content(self):
        self.content=self.content+self.response_content+"\n"
        return self.content,"",""

class assistant_for_modify(llm_model):
    def __init__(self):
        super().__init__()

    def clear_history(self):
        pass

class chat_functions(llm_model):
    # 模型选择调用，上面的也可以更新，懒得弄了
    def chat_fn(self,model):
        if model in self.model_qwen_list:
            model_function=self.qwen_model
        elif model in self.model_qianfan_list:
            model_function=self.qianfan_model
        elif model in self.model_zhipuai_list:
            model_function = self.chatglm_model
        elif model in self.model_kimi_list:
            model_function=self.kimi_model
        elif model in self.model_sparkai_list:
            model_function=self.sparkai_model
        elif model in self.model_deepseek_list:
            model_function=self.deepseek_model
        return model_function

class finetune_generation(chat_functions):
    def __init__(self):
        super().__init__()
        self.finetune_data = []         #未经过高贵的裁判模型
        self.show_data=[]
        self.finetune_datasets= []      #经过高贵的裁判模型
        self.num_select= int
    # 清除一切！！！
    def clear_all(self):
        self.finetune_data = []
        self.show_data=[]
        self.finetune_datasets = []
        self.num_select = 0
        print("清空所有数据,当前数据为空")
        return "",[[]],"",gr.Dropdown(choices=[], multiselect=True, label="选择存储的数据集编号(多选)",value="")
    # 对切分之后的数据进行处理
    def rebuild_question_and_answer(self,question, answer):
        if "]" in question:
            question = question.replace("]", "")
        if "[" in question:
            question = question.replace("[", "")
        if "]" in answer:
            answer = answer.replace("]", "")
        if "[" in answer:
            answer = answer.replace("[", "")
        return question, answer

    # 切分QA，采用正则表达式
    def question_answer(self,response):
        if type(response) == str:
            response=response.replace("\n","")
        questions=re.findall(r"用户：(.*?)回答",response)
        answers=re.findall(r"回答：(.*?)END",response)
        num=len(questions)
        for i in range(num):
            if len(questions) != len(answers):
                print("出现问题个数与回答个数不匹配的现象！")
                break
            question=questions[i]
            answer=answers[i]
            question,answer=self.rebuild_question_and_answer(question,answer)
            self.finetune_data.append({"question":question,"answer":answer})
    # 裁判模型，这里采用了deepseek模型，因为便宜！
    def judgement_model(self,judgement_prompt):
        model=""
        judgement_result=self.deepseek_model(model,judgement_prompt)
        if "1" in judgement_result:
            return 1
        elif "0" in judgement_result:
            return 0
    # 文本生成
    def generation(self,model,finetune_prompt):
        model_function=self.chat_fn(model)
        response=model_function(model,finetune_prompt)
        # 对生成的文本块进行处理
        self.question_answer(response)

    def generate_content(self,models,finetune_language,section,judgement):
        if models == []:
            print("当前没有选择模型，请重新选择模型！")
            return [[]],"请选择模型！！！"
        if section == "":
            print("当前没有输入文本，请输入文本！")
            return [[]],"当前没有输入文本，请输入文本！"
        print(f"当前选择的模型是：{models}，选择语言为：{finetune_language}，选择文本为：{section}，是否进行数据集筛选：{judgement}")
        finetune_prompt=f"请用{finetune_language}来生成对话数据集。请根据以下文本：{section}，请你为我构造多个单轮问答数据，包含用户的问题和你认为的标准答案。格式为：用户：[]。回答：[]END。"
        finetune_prompt=[{"role":"user","content":finetune_prompt}]
        # 将所有的数据集存储到了self.finetune_data=[{},{},{}]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for model in models:
                futures=[executor.submit(self.generation,model,finetune_prompt)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
        # 判断是否采用裁判模型筛选数据
        if judgement=="是":
            print("正在进行数据筛选...")
            print(self.finetune_data)
            print(len(self.finetune_data))
            # 监测数据是否异常
            if len(self.finetune_data) == 0:
                return [[]],"出现错误，请重新生成数据！"
            # 历遍数据，采用裁判模型筛选数据
            for i in range(len(self.finetune_data)):
                question=self.finetune_data[i]["question"]
                answer=self.finetune_data[i]["answer"]
                print(f"当前的问题：{question}，回答：{answer}")
                judgement_prompt=f"原文引用：【{section}】。问题：【{question}】;答案：【{answer}】。请分析判断这段回答是否符合原文引用，对于每一个问题和答案，先判断答案内容的准确性，如果符合，请回答1，不符合请回答0，无需做出推理。"
                judgement_prompt=[{"role":"user","content":judgement_prompt}]
                judgement_result=self.judgement_model(judgement_prompt)
                if judgement_result==0:
                    continue
                else:
                    self.show_data.append([i,question,answer])
                    self.finetune_datasets.append(self.finetune_data[i])
            print("结束数据筛选...")
            generation_codition=f"生成总数目：{len(self.finetune_data)}"+"\n"+f"筛选后总数目：{len(self.finetune_datasets)}"+"\n"+f"筛选率：{100*len(self.finetune_datasets)/len(self.finetune_data)}%"
            self.num_select=len(self.finetune_datasets)
            return self.show_data,generation_codition
        else:
            # 判断是否出现异常
            if len(self.finetune_data) == 0:
                return [[]],"出现错误，请重新生成数据！"
            # 历遍数据写入到表格
            for i in range(len(self.finetune_data)):
                question=self.finetune_data[i]["question"]
                answer=self.finetune_data[i]["answer"]
                self.show_data.append([i,question,answer])
            self.finetune_datasets=self.finetune_data
            print("无需进行数据筛选，直接将数据呈现给用户！")
            generation_codition=f"生成总数目：{len(self.finetune_data)}"
            self.num_select = len(self.finetune_datasets)
            return self.show_data,generation_codition
    # 保存数据！
    def save_finetune_data_csv(self,num):
        with open ("finetune_data_csv.csv","a",newline="") as file:
            writer=csv.writer(file)
            writer.writerow([self.finetune_datasets[num]["question"],self.finetune_datasets[num]["answer"]])
    def save_finetune_data(self,nums,finetune_type,finetune_data_condition):
        # 写入文件！
        with open(f"./finetune_data_{finetune_type}.json","a",encoding="utf-8") as file:
            try:
                data=json.load(file)
            except:
                data=[]
        with open(f"./finetune_data_{finetune_type}.json","w",encoding="utf-8") as file:
            for i in nums:
                if finetune_type == "llama-factory":
                    data.append({"instruction":self.finetune_datasets[i]["question"],"input":"","output":self.finetune_datasets[i]["answer"]})
                elif finetune_type == "chatglm3":
                    data.append({"conservations":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":self.finetune_datasets[i]["question"]},{"role":"assistant","content":self.finetune_datasets[i]["answer"]}]})
                self.save_finetune_data_csv(i)      #保存一份csv格式的
            json.dump(data,file,ensure_ascii=False)
        print("完成写入！")
        return finetune_data_condition+"\n"+"完成写入！"
    # 返回给复选框
    def dropdown_change(self):
        return gr.Dropdown(choices=[i for i in range(self.num_select)], multiselect=True, label="选择存储的数据集编号(多选)")