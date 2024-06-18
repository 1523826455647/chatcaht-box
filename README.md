使用教程：
首先在llm_list.py中填入你们的api，建议都填入。
接着pip install -r requirements.txt安装依赖。
最后点击run.py即可运行。

#模型添加
如果以后需要加入更多的模型，请参考如下的操作指南：
##阿里云：
直接在interaction_layout.py中加入模型，并且在llm_list.py中加入模型，在对应的栏目中加入即可，阿里云在这里命名了一个model_qwen_list列表，以后会考虑将两个结合起来。

##百度云：
命名了model_qianfan列表，同阿里云操作，但是在llm_list.py中的model_qianfan_dir（第44行代码）中加入相关的模型后缀，并且有的模型比较特殊，好像需要api啥的，也没搞懂怎么调用，但问题不大。这里表扬阿里云的简便操。

##deepseek目前只有一个模型，所以暂时加不了太多

##暗之月面：
操作同理阿里云

##智谱ai：
同阿里云操作

#对话
##对话前须知
1.在对话开始之前，你需要确保阿里云和百度云所有模型的计费都是开启的状态，特别注意这一点！！！
2.目前处理星火大模型，其他大模型都支持多轮对话，你要注意的是，如果多轮对话超过了api所规定的长度，会出现无法回答的问题！尤其是阿里云的max模型，仅支持6k的对话，如果需要长对话，请使用chatglm4和暗之月面32k和128k以及百度的模型。
3.所有对话历史都需要你亲自的去保存，应为我们无法知道用户是否需要保存这个历史，删除历史只需要选择对应的历史，点击删除历史按钮即可。历史将会保存在chat_history.json文件中。
4.保存历史之前可以先设置本次会话主题，当然你可以忽略这一点，我们会选择你对话的前五个词作为本次问答的主题。
5.对话之前可以先设置大语言模型的身份，当然你可以不设置，这里默认设置了几个身份，你可以自主选择。
6.如果你需要设置新的身份，请在interaction_layout.py中设置。
7.因为不会流式输出怎么弄，目前大模型都是直接返回所有问答，以后会学习一下如何添加这个功能，也希望各位能提供一些指导建议，谢谢！

##文章生成助手
1.在生成文章之前，你需要提供你的文章标题，llm会根据你的文章标题生成一个文章大纲，你需要复制每一章的标题到相应栏目，来生成文章，当然你也可以改变每一章的标题。
2.保存的文章都会在“最终作品窗口”处。

##微调数据集生成和助手
1.使用前请参考页面上方的提示。
2.切勿在生成微调数据集之后，在未清空的情况下，再次生成微调数据集，会出现一些错误。
3.生成微调数据集后，你所选择保存的数据集会保存在finetune_data_{数据保存类型}.json文件中，当然也会额外保存一份csv格式的数据集，方便你们换成其他数据格式，这里需要自行编写代码。

#未来开发
1.或许会在微调数据集生成中加入更多的功能，现在是根据文本文章来生成对话数据集，以后可能会加入根据问答数据集来生成问答数据集，或者加入新的东西，比如对于文章生成等等的微调数据。
2.接入更多的大模型，这一步你们可以自己完成，当然我也会完善这个，或许会接入豆包大模型等等。
3.图片生成和图片理解的大模型也许也会加入。

#最后
1.由于本人不是计算机专业的，只是一个卑微的土木鼠，写的代码估计也不是非常的好，希望见谅，多给我提出宝贵的建议！
2.后续的一些功能我会尽可能的去完善，也希望给大家一个好的用户体验！
3.如果遇到一些bug之类的，我们可以在issue中讨论！
4.如果可以的话，点亮上方的小星星给予我一定的创作动力，谢谢！


