from openai import OpenAI
import numpy as np
import requests
import torch
import cv2
import random
import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline
import requests
from PIL import Image as Img
from PIL import ImageTk
from tkinter import dialog
import tkinter.font as tkFont
from tkinter import *           # 导入 Tkinter 库

import threading
# Python3.x 导入方法
#from tkinter import *

prompt_template = [
    'Indie game art, Vector Art, Borderlands style, Arcane style, Cartoon style, Line art, Disctinct features, Hand drawn, Technical illustration, Graphic design, Vector graphics, High contrast, Precision artwork, Linear compositions, Scalable artwork, Digital art, cinematic sensual, Sharp focus, humorous illustration, big depth of field, Masterpiece, trending on artstation, Vivid colors, trending on ArtStation, trending on CGSociety, Intricate, Low Detail, dramatic',
    'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured',
    'vibrant, cute, cartoony, fantasy, playful,',
    'bright, cheerful, light-hearted, cartoonish, cute',
    '3D cartoon, colorful, Fortnite Art Style',
    'cartoon, illustration, painting, frame',
]


def TTS(prompt):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/ThT5KcBeYPX3keUQqHPh"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "35dea74f165d365f5eef7685fbb82441"
    }

    data = {
        "text": prompt,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)


client = OpenAI(api_key='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')


def translate_prompt(context, prompt):
    content1 = "You are an assistant who is good at writing DALL-E 3 prompts. The story context and previous prompts are{}".format(
        context)
    content2 = """Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is XiaoNiu, 
  3. XiaoNiu is with features: a cute boy, red hair, green eyes, dressed in colorful attire including playful raincoats and rain boots.cartoon style, illustration, painting, frame,
  4. The character in each prompt should have the same appearance,
  5, The output images should not contain any words,
  """.format(prompt)
    messages = [
        {"role": "system", "content": content1},
        {"role": "user", "content": content2}
    ]
    trans_completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    return trans_completion.choices[0].message.content


'''
####chibi####
Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is "chibi", and the converted prompt should show chibi,
  3. The picture is anime style, cartoon style,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####ying####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is YING GIRL, and the converted prompt should show YING GIRL, special clothing, blonde hair, 
  3. ying girl is with special clothing, blonde hairle, cartoon style,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####keqing####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is keqing (genshin impact), and the converted prompt should show keqing (genshin impact), purple hair,twintails, 
  3. keqing (genshin impact) is with purple hair,twintails, black dress,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####ningguang####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is ningguang, and the converted prompt should show ningguang, Chinese clothing, bare shoulders, red eyes,
  3. Ningguang is with Chinese clothing, bare shoulders, red eyes,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####shenlininghua####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is shenlininghua, and the converted prompt should show shenlininghua,
  3. No text should appear in the picture,
  4. The character in each prompt should have the same appearance,
'''

'''
####teddy####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is teddy, and the converted prompt should show teddy,
  3. No text should appear in the picture,
  4. The character in each prompt should have the same appearance,
'''

'''
####PEPE####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is PEPE, and the converted prompt should show PEPE,
  3. The image is cartoon style,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####teletubbiesxl####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is teletubbiesxl, and the converted prompt should show teletubbiesxl,
  3. The image is cartoon style,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####aoki####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is aoki, and the converted prompt should show aoki, blonde, boy,
  3. aoki is with blonde hair,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''

'''
####OHXW####
  Convert the following sentence into English prompt:{}\n, the requirements are as follows:
  1. The prompt must be less than 77 tokens,
  2. The protagonist of the story is OHXW, and the converted prompt should show OHXW, star12, red hair, green eyes,
  3. OHXW is with star12, red hair, green eyes, cartoon style,
  4. No text should appear in the picture,
  5. The character in each prompt should have the same appearance,
'''


def expand_prompt(prompt):
    content1 = "You are good at expanding sentences to describe details in more detail, and the expansion result should not exceed 100 tokens."
    content2 = "Write this sentence in more detail:{}".format(prompt)
    messages = [
        {"role": "system", "content": content1},
        {"role": "user", "content": content2}
    ]
    trans_completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    return trans_completion.choices[0].message.content


class Conversation:
    def __init__(self, prompt, round):
        self.prompt = prompt
        self.round = round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})
        self.information = ''

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=self.messages,
                temperature=0.5,
                max_tokens=2048,
                top_p=1,
            )
        except Exception as e:
            print(e)
            return e
        message = response.choices[0].message.content
        # message = expand_prompt(message)
        self.messages.append({"role": "assistant", "content": message})
        # self.information = self.information+message
        if len(self.messages) > self.round * 2 + 1:
            text = self._build_message(self.messages)
            # print (text)
            # print ("=====summarize=====")
            summarize = self.summarize(text)
            # print (summarize)
            # print ("=====summarize=====")
            self.messages = []
            self.messages.append({"role": "system", "content": summarize})
        return message

    def summarize(self, text, max_tokens=200):
        ms = [{"role": "system", "content": text + "\n\nplease summerize：\n"}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=ms,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def translate(self, text, max_tokens=500):
        content1 = "You are good at translate sentences into Chinese."
        content2 = """Translate thess sentence into Chinese:{}\n, the requirements are as follows:
        1. The prompt must be less than 200 tokens,
        2. The answer should only contain the translate result, 
        """.format(prompt)
        ms = [
            {"role": "system", "content": content1},
            {"role": "user", "content": content2}
        ]
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=ms,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _build_message(self, messages):
        text = ""
        for message in messages:
            if message["role"] == "user":
                text += "User : " + message["content"] + "\n\n"
            if message["role"] == "assistant":
                text += "Assistant : " + message["content"] + "\n\n"
        return text

    def append_start(self, choice):
        self.information = self.information + choice + ':'

    def append_prompt(self, prompt):
        self.information = self.information + prompt

    def generate_images(self, prompt):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = response.data[0].url
        response = requests.get(url, stream=True)
        img = Img.open(response.raw).convert("RGB")
        # import pdb; pdb.set_trace()
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img


# MODEL_DIR = "./lora/Designer_BlindBox-000015.safetensors"
# MODEL_DIR = "./lora/keqing3.safetensors"
# MODEL_DIR = "./lora/ningguang2x_xl.safetensors"
# MODEL_DIR = "./lora/aoki_xl_v10.safetensors"
# MODEL_DIR = "./lora/DD-pepe-v2.safetensors"
# MODEL_DIR = "./lora/teletubbiesxl.safetensors"
# MODEL_DIR = "./lora/Shin-chan001.safetensors"
MODEL_DIR = "./lora/1star-000004.safetensors"

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
# #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# #ddim = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
# pipe = pipe.to("cuda")
# pipe.load_lora_weights(MODEL_DIR)
# #pipe.enable_xformers_memory_efficient_attention()
#
# def generate_images(prompt):
#     image = pipe(
#         prompt,
#         num_inference_steps=30,
#         guidance_scale=7.5,
#     ).images[0]
#     return image
cprompt = """
您是一位擅长为尝试学习自然科学的孩子设计长故事板的老师。故事必须尽可能具有教育意义，并且采用第一人称。你应该站在孩子受教育的角度去讲，给孩子起一个名字：小牛。您的答案需要满足以下要求：
1. 您的回答必须是中文。每个答案都是一帧，每个答案必须少于 77 个token。答案必须易于 5 岁儿童理解；
2. 对于6岁以下的儿童，句子应尽可能简单；
3、如果故事结束了，回答“故事结束了”；
4、故事尽量长；
5. 首先设置故事场景。详细描述场景，包括地点、时间段以及任何相关的气氛或环境细节；
6. 每个故事仅介绍一次主要角色，提供有关他们的性格、外貌和背景的信息；
7. 概述主角面临的中心冲突或问题。这可能是内心的斗争，与另一个角色的冲突，或者环境带来的挑战；
8. 描述推动故事向前发展的一系列事件。这些应该是主角采取的行动或发生在他们身上的事件，导致高潮；
9. 将故事引向高潮，即故事最激烈的时刻，主角面临最大的挑战或做出关键的决定；
10. 通过解决冲突来结束故事。描述主角如何改变或者他们从经历中学到了什么；
11. 以给读者留下深刻印象的结尾结尾。这可能是一个最后的转折，一个结论性的陈述，或者一个萦绕在脑海中的开放式问题；
12. 或者，您可以要求特定的风格元素，例如幽默的语气、诗意的语言或悬疑的节奏；
13. 融入揭示人物特征并推动情节的对话。对话可以成为展示人物关系和动态的有力工具；
14. 一旦你收到“你应该在下一帧为用户设计交互”，你应该设计两个选择，交互一定要更自然。
15. 故事不需要写出第几帧。
16. 故事不需要写出“你应该在下一帧为用户设计交互”这句话，同时如果没有收到“你应该在下一帧为用户设计交互”则不要设计选择 """

dgPrompt = """
您是一位擅长为尝试学习自然科学的孩子设计课程大纲的老师。您的答案需要满足以下要求：
1. 您的回答必须是中文。大纲内容必须适合5岁儿童学习；
2. 大纲需要分为多个章节，每个章节需要包含几个相关的小节；
3、对于6岁以下的儿童，大纲每一小节的内容应尽可能简单；
4、回答仅需包括大纲内容即可；
"""

prompt = """You are a teacher who is good at designing storyboards for long stories for children who trying to learn english. The story needs to be educational, as long as possible, with first person. You should tell it from the perspective of the child to be educated and give the child a name: oxhw. Your answer needs to meet the following requirements:
1. Your answer must be in English. Each answer is one frame, each answer must be less than 77 tokens. Answers must be easy for children aged 3 to understand; 
2. The word frame should not appear in the answers;
3. The sentences should be AS SIMPLE AS POSSIBLE for children aged less than 6;
4. If the story is over, answer, "The story is over";
5. The story should be as long as possible;
6. Begin by setting the scene for the story. Describe the setting in detail, including the location, time period, and any relevant atmosphere or environmental details;
7. Introduce the main character(s) only once per story, providing information about their personality, appearance, and background;
8. Outline the central conflict or problem that the main character faces. This could be an internal struggle, a conflict with another character, or a challenge posed by the environment;
9. Describe a series of events that move the story forward. These should be actions taken by the main character or events that happen to them, leading to a climax;
10. Lead the story to a climax, the most intense point of the story, where the main character faces their biggest challenge or makes a crucial decision;
11. Conclude the story by resolving the conflict. Describe how the main character has changed or what they have learned from their experiences;
12. End with a closing that leaves a lasting impression on the reader. This could be a final twist, a conclusive statement, or an open-ended question that lingers in the mind;
13. Optionally, you can ask for specific stylistic elements, like a humorous tone, poetic language, or suspenseful pacing;
14. Incorporate dialogue that reveals character traits and advances the plot. Dialogue can be a powerful tool to show character relationships and dynamics;
15. You should design two choices once receiving "You should have an interaction with the user in the next frame", the interaction must be more natural.
"""

# 3. 故事的剧情尽量长一些；

img = []
def runAI():
    t1 = threading.Thread(target=start)
    t1.start()

def runDG():
    t1 = threading.Thread(target=startdg)
    t1.start()
def startdg():
    message = "生成中。。。"
    label_1 = Label(root, text=message, wraplength=1000, font=ft)
    label_1.pack()
    conv = Conversation(dgPrompt, 100)
    choice = "你需要教一个小孩学习什么是 " + e.get()
    message = conv.ask("标题是 {}".format(
        choice) + "请你设计大纲")
    label_1 = Label(root, text=message, wraplength=1000, font=ft)
    label_1.pack()
    frame.update()
def start():
    conv = Conversation(cprompt, 100)
    options = ""
    round = 0
    interact = False
    while True:
        round += 1
        rand = random.random()
        if round == 1:
            print("你希望是怎样的故事")
            choice = "你需要教一个小孩学习什么是 " + "家中的宠物，猫咪和小狗"
            # print("What kind of story do you want?")
            # choice = input() + ", You need to teach a child to learn the letter 'A'"
            # choice = translate_prompt(conv.information, choice)

            conv.append_start(choice)
            message = conv.ask("标题是 {}".format(
                choice) + "请你设计故事和内容, 然后下一帧是")
            if "故事结束了" not in message:
                # message = expand_prompt(message)
                # try:
                conv.append_prompt(message)
                label_1 = Label(root, text=message, wraplength=1000, font=ft)
                label_1.pack()

                print(message)
                message = translate_prompt(conv.information, message)
                # try:
                image = conv.generate_images(message)

                image.save('ohxw_images/{}.png'.format(round))
                img_open = Img.open('ohxw_images/{}.png'.format(round))
                img_png = ImageTk.PhotoImage(img_open)
                label_img = Label(root, image=img_png)
                img.append(img_png)
                label_img.pack()
                frame.update()

                # except:
                #    print("Error")
                #    continue
        elif interact:
            d = dialog.Dialog(root
                              , {'title': '选择',  # 标题
                                 'text': options,  # 内容
                                 'bitmap': 'question',  # 图标
                                 'default': 0,  # 设置默认选中项
                                 # strings选项用于设置按钮
                                 'strings': ('选项1',
                                             '选项2',)})
            choice = "用户的选择是:" + str(d.num + 1)
            choice += ", 下一帧是"
            print("Q:" + choice)
            message = conv.ask(choice)
            if "故事结束了" not in message:
                # message = expand_prompt(message)
                print(message)
                label_1 = Label(root, text=message, wraplength=1000, font=ft)

                label_1.pack()
                conv.append_prompt(message)
                message = translate_prompt(conv.information, message)
                try:
                    image = conv.generate_images(message)
                    image.save('ohxw_images/{}.png'.format(round))
                    img_open = Img.open('ohxw_images/{}.png'.format(round))
                    img_png = ImageTk.PhotoImage(img_open)
                    label_img = Label(root, image=img_png)
                    img.append(img_png)
                    label_img.pack()
                    frame.update()

                except:
                    print("Error")
                    continue
                interact = False
        elif not interact and rand < 0.5 and round >= 3:
            choice = "你应该在下一帧为用户设计交互. 交互需要更加自然且包含两个选择, 另外交互的格式是 1: xxx, 2: xxx.\n根据上述要求设计下一帧"
            print("Q:" + choice)
            message = conv.ask(choice)
            # conv.append_prompt(message)
            if "故事结束了" not in message:
                # message = expand_prompt(message)
                print(message)
                options = message
                label_1 = Label(root, text=message, wraplength=1000, font=ft)

                label_1.pack()
                interact = True
        else:
            choice = "并且下一帧是"
            print("Q:" + choice)
            message = conv.ask(choice)
            if "故事结束了" not in message:
                # message = expand_prompt(message)
                print(message)
                label_1 = Label(root, text=message, wraplength=1000, font=ft)
                label_1.pack()
                conv.append_prompt(message)
                message = translate_prompt(conv.information, message)
                try:
                    image = conv.generate_images(message)
                    image.save('ohxw_images/{}.png'.format(round))
                    img_open = Img.open('ohxw_images/{}.png'.format(round))
                    img_png = ImageTk.PhotoImage(img_open)
                    label_img = Label(root, image=img_png)
                    img.append(img_png)
                    label_img.pack()
                    frame.update()

                except:
                    print("检测到异常信息，这步不出图")
                    continue
        if "故事结束了" in message:
            label_1 = Label(root, text=message, wraplength=1000, font=ft)
            label_1.config(bg="#8c52ff")
            label_1.pack()
            print(message)
            break
# 创建一个 Canvas 窗口
frame = Tk()
frame.title('儿童教育故事书')        #窗口标题
canvas = Canvas(frame)
canvas.pack(side='left', fill='both', expand=True)

# 添加 Scrollbar 窗口
scrollbar = Scrollbar(frame, orient='vertical', command=canvas.yview)
scrollbar.set(0.5, 1)
scrollbar.pack(side='right', fill='y')


# 绑定 Scrollbar 和 Canvas 窗口
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

# 创建一个 Frame 并把它添加到 Canvas 中
root = Frame(canvas)

ft = tkFont.Font(family='华文隶书', size=20,weight=tkFont.BOLD,slant=tkFont.ITALIC,\
             underline=1,overstrike=1)
canvas.create_window((0, 0), window=root, anchor='nw')
# label_1 = Label(root, text="你需要教一个小孩学习什么是", wraplength=1000, font=ft)
#
# label_1.pack()
# e = StringVar()
# # 使用textvariable属性，绑定字符串变量e
# entry = Entry(root,textvariable = e)
# e.set('请输入……')
# entry.pack()
btn = Button(root,text ="开始",command = runAI)
btn.pack()
root.mainloop()

