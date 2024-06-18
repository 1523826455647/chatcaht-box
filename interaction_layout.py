import csv
import json
import time
import gradio as gr
from brain_layout import model_chat,writer_expert,finetune_generation

model_qwen=["qwen-turbo","qwen-long","qwen-plus","qwen-max","qwen-max-0428","qwen-max-0403","qwen-max-0107","qwen-7b-chat","qwen-14b-chat","qwen-72b-chat","qwen1.5-7b-chat","qwen1.5-14b-chat","qwen1.5-72b-chat","qwen1.5-110b-chat"]
model_qianfan=["ERNIE-3.5-8K","ERNIE-3.5-128K","ERNIE-3.5-8K-0205","ERNIE-4.0-8K","ERNIE-4.0-8K-0104","ERNIE-4.0-8K-0329","ERNIE-3.5-8K-0205","ERNIE-Speed-128K(预览版)"]
model_zhipuai=["glm-4","glm-3-turbo","glm-4-0520","glm-4-air","glm-4-airx","glm-4-flash"]
model_kimi=["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k"]
model_sparkai=["星火大模型v3.5","星火大模型v3.0"]
model_deepseek=["deepseek-chat"]

model_list=model_qwen+model_qianfan+model_zhipuai+model_kimi+model_sparkai+model_deepseek

class page_edit():
    def __init__(self):
        super().__init__()
        self.history_list=[]
    def change_expert(self,expert):
        if expert == "AI写作导师":
            prompt="我想让你做一个AI写作导师。我将为您提供一名需要帮助改进其写作的学生，您的任务是使用人工智能工具（例如自然语言处理）向学生提供有关如何改进其作文的反馈。您还应该利用您在有效写作技巧方面的修辞知识和经验来建议学生可以更好地以书面形式表达他们的想法和想法的方法。"
        elif expert == '自定义':
            prompt=""
        elif expert == "自然语言处理专家":
            prompt="作为自然语言处理助手，你的主要任务是帮助我了解和掌握自然语言处理（NLP）的知识。你可以为您提供关于NLP的基本概念、技术原理、应用场景以及最新进展的详细解释和指导。无论我是初学者还是有经验的开发者，你都会尽力为您提供个性化、高效的学习和支持。"
        elif expert == "python大师":
            prompt="作为Python大师，你的主要任务是帮助我学习和掌握Python编程语言。你可以为我提供关于Python的基本语法、数据结构、函数、类、模块、库等方面的详细解释和指导。无论我是初学者还是有经验的开发者，你都会尽力为我提供个性化、高效的学习和支持。"
        return gr.Textbox(value=prompt,label="设定llm回答的身份",max_lines=12,lines=12)

    def change_model(self,model):
        print(f"已经切换成：{model}")
        return gr.Chatbot(label="欢迎使用认知多模型",height=624)

    def clear_prompt(self):
        return "","自定义"

class interaction_layout(page_edit):
    def __init__(self):
        super().__init__()
        self.reload_history_list()
        self.finetune_list=[]
    def load_history_list(self):
        with open("chat_history.json", "r", encoding="utf-8-sig") as file:
            history_list=[]
            try:
                data = json.load(file)
            except:
                data=[]
            for item in data:
                key=list(item.keys())
                history_list.append(key[0])
            print(history_list)
        return history_list

    def reload_history_list(self):
        time.sleep(0.9)
        self.history_list=self.load_history_list()
        return gr.Dropdown(choices=self.history_list,value="",allow_custom_value=True)

    def after_save_history(self,theme):
        time.sleep(1)
        return gr.Dropdown(choices=self.history_list,value=theme,allow_custom_value=True)

    def main_page(self):
        with gr.Blocks() as demo:
            with gr.Tab("认知多模型"):
                with gr.Row():
                    with gr.Column(min_width=200):
                        # 参数设置区域(设置一个prompt区？身份区域，预设几个身份)
                        model_select=gr.Dropdown(model_list,label="模型选择",value="glm-3-turbo")
                        prompt_expert=gr.Textbox(label="设定llm回答的身份",max_lines=12,lines=12)
                        expert=gr.Dropdown(["自定义","AI写作导师","自然语言处理专家","python大师"],value="自定义",label="身份设定",info="只需设定一次即可")
                        theme = gr.Textbox(label="本次聊天主题，用于保存历史")
                        self.history_dropdown=gr.Dropdown(choices=self.history_list)
                        with gr.Row():
                            history_reload_btn=gr.Button("刷新历史",scale=1,)
                            del_history_btn=gr.ClearButton(components=[theme],value="删除历史",scale=1,)
                    with gr.Column(min_width=1200):
                        # 交流区
                        output_text=gr.Chatbot(label="欢迎使用认知多模型",height=624)
                        input_text=gr.Textbox(label="输入框",lines=4,max_lines=4)
                        # 交互式按钮
                        with gr.Row():
                            text_btn_submit = gr.Button("提交", scale=0, min_width=600)
                            text_btn_save_history=gr.Button("保存历史")
                            text_btn_clear = gr.ClearButton([output_text, input_text,theme,self.history_dropdown], value="清除聊天记录",
                                                            min_width=170)
            with gr.Tab("文章生成助手"):
                with gr.Row():
                    with gr.Column(min_width=200):
                        title_text=gr.Textbox(label="文章标题",lines=1)
                        model_select_writer=gr.Dropdown(choices=["qwen-max","ERNIE-3.5-128K","moonshot-v1-32k","moonshot-v1-128k","deepseek-chat","glm-3-turbo","glm-4"])
                        writer_type=gr.Dropdown(label="写作风格",choices=["报告性写作","议论性写作","叙事性写作","描述性写作"])
                        struct_text=gr.Textbox(label="文章大纲",lines=20,max_lines=20)
                        title_btn_submit=gr.Button("提交标题/重写")
                        title_btn_clear=gr.ClearButton([title_text,struct_text,writer_type],value="清除一切")
                    with gr.Column(min_width=1200):
                        with gr.Tab("对话窗口"):
                            with gr.Column():
                                section_box=gr.Textbox(label="当前写作的章节")
                                content_box=gr.Textbox(label="文本生成框",lines=30,max_lines=30)
                            with gr.Row():
                                content_write_btn=gr.Button("开始写这个章节")
                                content_submit_btn=gr.Button("提交该段的文章")
                                rewrite_content_btn=gr.Button("AI重写该段文章")

                        with gr.Tab("最终作品窗口"):
                            with gr.Column():
                                finish_content=gr.Textbox(label="最终文章",lines=35,max_lines=35)
            with gr.Tab("文章优化助手"):
                with gr.Row():
                    with gr.Column():
                        model_select_assistant=gr.Dropdown(model_list)
                        suggestion_for_llm=gr.Textbox(label="给出你想该文章如何修改")
                        perset_suggestion=gr.Dropdown(["xxxx"])
                    with gr.Column():
                        # 交流区
                        output_text_assistant = gr.Chatbot(label="欢迎使用认知多模型", height=624)
                        input_text_assistant = gr.Textbox(label="输入框", lines=4, max_lines=4)
                        # 交互式按钮
                        with gr.Row():
                            text_btn_submit_assistant = gr.Button("提交", scale=0, min_width=600)
                            text_btn_clear_assistant = gr.ClearButton([output_text, input_text], value="清除聊天记录",
                                                            min_width=170)
            with gr.Tab("知识库问答助手"):
                with gr.Row():
                    with gr.Column():
                        # 参数设置区
                        pass
                    with gr.Column():
                        # 交互区
                        pass
                    with gr.Column():
                        # 知识库匹配区
                        pass
            with gr.Tab("文档问答助手"):
                with gr.Row():
                    with gr.Column():
                        # 输入区域
                        # 参数设置
                        pass
                    with gr.Column():
                        # 交互区
                        pass
            with gr.Tab("微调数据集生成助手"):
                with gr.Row():
                    with gr.Column():
                        # 模型选择&参数设置&数据格式
                        welcome_text="欢迎使用微调数据集工厂-手动版，作者：笨鼠鼠"+"\n"+"使用说明："+"\n"+"选择模型-->选择参数-->输入文本-->提交-->等待-->刷新微调数据集选择列表-->选择数据-->提交数据-->清除一切"+"\n"+"如果需要该保存路径，从brain_layout.py(lines=329&334)中修改！"
                        gr.Textbox(lines=4,max_lines=4,label="功能介绍",interactive=False,value=welcome_text)
                        model_select_finetune=gr.Dropdown(model_list,multiselect=True,label="选择模型")
                        with gr.Row():
                            finetune_language=gr.Radio(["中文","英文"],label="选择语言",value="中文",info="微调数据集的语种",scale=2)
                            judgement=gr.Radio(["是","否"],label="是否需要启动裁判模型",value="否",info="该功能会消耗大量token",scale=2)
                            finrtune_data_type=gr.Radio(["llama-factory","chatglm3"],value="llama-factory",label="数据集格式",info="最好用llama-factory",scale=3)
                        text_for_finetune_text_data=gr.Textbox(label="输入文本",lines=15,max_lines=15)
                        with gr.Row():
                            submit_btn_finetune_text_data=gr.Button("提交")
                            clear_btn_finetune_text_data=gr.ClearButton(value="清除一切")
                    with gr.Column():
                        # 微调数据集生成取
                        with gr.Column():
                            data=gr.DataFrame(label="微调数据集",interactive=False,wrap=True,headers=["idx","question","answer"],height=475)
                            finetune_data_condition=gr.Textbox(label="程序运行状态栏",lines=3,max_lines=3)
                            select_finetune_data=gr.Dropdown(choices=self.finetune_list,multiselect=True,label="选择存储的数据集编号(多选)")
                            with gr.Row():
                                renew_select_finetune=gr.Button("刷新微调数据集选择列表")
                                submit_finetune_data_btn=gr.Button("提交/保存微调数据")
            # ***************************认知多模型*****************************************
            expert.change(fn=self.change_expert,inputs=expert,outputs=prompt_expert)
            model_select.select(fn=self.change_model,inputs=model_select,outputs=output_text)

            chat_with_model=model_chat()
            text_btn_submit.click(fn=chat_with_model.chat_with_llm,inputs=[model_select,input_text,prompt_expert,output_text],outputs=[input_text,output_text])
            text_btn_submit.click(fn=self.clear_prompt,outputs=[prompt_expert,expert])

            text_btn_save_history.click(fn=chat_with_model.save_history,inputs=theme,outputs=theme)
            text_btn_save_history.click(fn=self.reload_history_list,outputs=self.history_dropdown)
            text_btn_save_history.click(fn=self.after_save_history,inputs=theme,outputs=self.history_dropdown)

            text_btn_clear.click(fn=chat_with_model.clear_chat_history,outputs=input_text)
            history_reload_btn.click(fn=self.reload_history_list,outputs=self.history_dropdown)
            del_history_btn.click(fn=chat_with_model.del_history,inputs=self.history_dropdown,outputs=output_text)

            del_history_btn.click(fn=self.reload_history_list,outputs=self.history_dropdown)
            model_select.select(fn=chat_with_model.clear_history, inputs=model_select, outputs=output_text)

            self.history_dropdown.select(fn=chat_with_model.renew_history_chat,inputs=self.history_dropdown,outputs=[output_text,theme])
            # ***************************文章生成助手*****************************************
            writer=writer_expert()

            title_btn_submit.click(fn=writer.clear_history)
            title_btn_submit.click(fn=writer.llm_struct,inputs=[model_select_writer,title_text,writer_type],outputs=struct_text)
            content_write_btn.click(fn=writer.generate_content,inputs=[model_select_writer,section_box],outputs=content_box)
            content_submit_btn.click(fn=writer.sumbit_content,outputs=[finish_content,content_box,section_box])
            rewrite_content_btn.click(fn=writer.rewriter_content,inputs=[model_select_writer,section_box],outputs=content_box)
            title_btn_clear.click(fn=writer.clear_all,outputs=[section_box,content_box,finish_content])

            # ***************************微调数据集生成助手*****************************************
            finetune=finetune_generation()
            submit_btn_finetune_text_data.click(fn=finetune.generate_content,inputs=[model_select_finetune,finetune_language,text_for_finetune_text_data,judgement],outputs=[data,finetune_data_condition])
            renew_select_finetune.click(fn=finetune.dropdown_change,outputs=select_finetune_data)
            clear_btn_finetune_text_data.click(fn=finetune.clear_all,outputs=[text_for_finetune_text_data,data,finetune_data_condition,select_finetune_data])
            submit_finetune_data_btn.click(fn=finetune.save_finetune_data,inputs=[select_finetune_data,finrtune_data_type,finetune_data_condition],outputs=finetune_data_condition)

        demo.launch()

if __name__ == "__main__":
    layout = interaction_layout()
    layout.main_page()
