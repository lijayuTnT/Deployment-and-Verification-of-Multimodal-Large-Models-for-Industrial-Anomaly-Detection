import argparse
import base64
import json

import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
import seaborn as sns
import logging
from openai import OpenAI
from Dinomaly.locate_anomaly import locate

sys.path.append("..")

from difflib import get_close_matches
from helper.summary import caculate_accuracy_mmad

cfg = {
    "data_path": "../../../dataset/MMAD",
    "json_path": "../../../dataset/MMAD/mmad.json",
    "knowledge_path": "../../../dataset/MMAD/domain_knowledge.json",
    "anomaly_path": "./anomaly_location"
}

instruction = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly.
'''

client = OpenAI(
    api_key = "sk-3f31b82e72ff4b9897c939d0e37f4f1a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class QwenQuery:

    def __init__(self, image_path, mask_path,text_gt,mask=True,RAG=True,knowledge=""):
        self.image_path = image_path
        self.mask_path = mask_path
        self.text_gt = text_gt
        self.max_image_size = (512, 512)
        self.mask= mask
        self.RAG = RAG
        self.knowledge = knowledge

    def encode_image_to_base64(self, image):
        # 获取图像的尺寸
        height, width = image.shape[:2]
        # 计算缩放比例
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)

        # 使用新的尺寸缩放图像
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')  

    def parse_conversation(self,text_gt):
        Question = []
        Answer = []
        Option = []
        # 想要匹配的关键字
        keyword = "conversation"

        # 遍历字典中的所有键
        for key in text_gt.keys():
            # 如果键以关键字开头
            if key.startswith(keyword):
                # 获取对应的值
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):
                    # 打乱选项的顺序
                    options_items = list(QA['Options'].items())
                    random.shuffle(options_items)  # 随机排序选项

                    # 重建选项文本并创建一个新的选项到答案的映射
                    options_text = ""
                    new_answer_key = None
                    for new_key, (original_key, value) in enumerate(options_items):
                        options_text += f"{chr(65 + new_key)}. {value}\n"  # 65是字母A的ASCII码
                        if QA['Answer'] == original_key:
                            new_answer_key = chr(65 + new_key)  # 更新答案的键
                    option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}

                    questions_text = QA['Question']
                    if QA['type'] == 'Anomaly Detection':
                        questions_text = "Is there any anomaly in the image?"
                        Question.append(
                            {
                                "type": "text",
                                "text": f"{questions_text} \n"
                                        f"{options_text}"
                            },
                        )
                    else:
                        Question.append(
                            {
                                "type": "text",
                                "text": f"{questions_text} \n"
                                        f"{options_text}"
                            },
                        )
                    Option.append(option_dict)
                    if new_answer_key is not None:
                        Answer.append(new_answer_key)
                    else:
                        raise ValueError("Answer key not found after shuffling options.")
                break
        return Question, Answer , Option

    def parse_answer(self, response_text, options=None):
        # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
        # pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
        pattern = re.compile(r'\b([A-E])\b')
        # 使用正则表达式提取答案
        answers = pattern.findall(response_text)

        if len(answers) == 0 and options is not None:
            print(f"Failed to extract answer from response: {response_text}")
            # 模糊匹配options字典来得到答案
            options_values = list(options.values())
            # 使用difflib.get_close_matches来找到最接近的匹配项
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                # 如果有匹配项，找到对应的键
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        answers.append(key)
                        break
        return answers

    def get_query(self, conversation):
        image = cv2.imread(self.image_path)
        base64_image = self.encode_image_to_base64(image)
        if self.mask:
            heat_map = cv2.imread(self.mask_path)
            base64_heat_map = self.encode_image_to_base64(heat_map)
            if self.RAG:
                messages = [
                {
                    "role": "system",
                    "content":[{"type":"text","text":instruction}]
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type":"text","text": "Answer with the option's letter from the given choices directly."
                        },
                        {
                            "type":"text","text":"Following is the query image:"
                        },
                        {
                            "type":"image_url",
                            "image_url":{"url":f"data:image/jpg;base64,{base64_image}"}
                        },
                        {
                            "type":"text","text":"Following is the domain knowledge corresponding to the image."
                        },
                        {
                            "type":"text","text":knowledge
                        },
                        {
                            "type":"text","text":"Following is a heatmap corresponding to the previous image. Possible anomalies areas are highlighted in orange-red,which you can use to answer the query questions. Since every heatmap has oranged-red areas,you should judge whether there is a defect in the query image before using the heatmap!"
                        },
                        {
                            "type":"image_url",
                            "image_url":{"url":f"data:image/jpg;base64,{base64_heat_map }"}
                        },
                        {
                            "type":"text","text": "Following are the questions: "
                        },
                    ]
                },
                {
                    "role": "user",
                    "content":conversation
                }
                    ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content":[{"type":"text","text":instruction}]
                    },
                    {
                        "role": "user",
                        "content":[
                            {
                                "type":"text","text": "Answer with the option's letter from the given choices directly."
                            },
                            {
                                "type":"text","text":"Following is the query image:"
                            },
                            {
                                "type":"image_url",
                                "image_url":{"url":f"data:image/jpg;base64,{base64_image}"}
                            },
                            {
                                "type":"text","text":"Following is a heatmap corresponding to the previous image. Possible anomalies areas are highlighted in orange-red,which you can use to answer the query questions. Since every heatmap has oranged-red areas,you should judge whether there is a defect in the query image before using the heatmap!"
                            },
                            {
                                "type":"image_url",
                                "image_url":{"url":f"data:image/jpg;base64,{base64_heat_map }"}
                            },
                            {
                                "type":"text","text": "Following are the questions: "
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content":conversation
                    }
                        ]
        # 构建查询
        else:
            messages =[
                {
                    "role": "system",
                    "content":[{"type":"text","text":instruction}]
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type":"text","text": "Answer with the option's letter from the given choices directly."
                        },
                        {
                            "type":"text","text":"Following is the query image:"
                        },
                        {
                            "type":"image_url",
                            "image_url":{"url":f"data:image/jpg;base64,{base64_image}"}
                        },
                        {
                            "type":"text","text": "Following are the questions: "
                        },
                    ]
                },
                {
                    "role": "user",
                    "content":conversation
                }
            ]
        return messages

    def generate_answer(self):
        questions, answers , options= self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        gpt_answers = []
        messages = []
        for i in range(len(questions)):
            question = questions[i:i+1]
            option = options[i]
            query = self.get_query(question)
            if i==0:
                messages=query
            else:
                messages.append(query[2])
            completion= client.chat.completions.create(model="qwen-vl-max-latest",messages=messages)
            assistant_message = completion.choices[0].message
            messages.append(assistant_message.model_dump())
            gpt_answer = self.parse_answer(completion.choices[0].message.content,option)
            if len(gpt_answer) == 0:
                logging.error(f"No matching answer at {self.image_path}: {question}")
            gpt_answers.append(gpt_answer[-1])

        return questions, answers, gpt_answers

def get_combined_and_good(file_path, item_category):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if item_category not in data.keys():
        raise ValueError(f"类别 '{item_category}' 不存在于数据中。")

    item_data = data[item_category]
    combined = item_data.get("combined", "")
    good = item_data.get("good", "")

    return combined+"\n"+good

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduce", type=bool, default = False)
    parser.add_argument("--mask", type=bool, default = False)
    parser.add_argument("--RAG", type=bool, default = False)
    parser.add_argument("--model_path", type=str, default="Qwen/qwen-vl-max-latest")
    parser.add_argument("--mask_encoding_way", type=str, default="base64")
    parser.add_argument("--model_iter",type=int, default=500)
    args = parser.parse_args()
    model_path = args.model_path
    model_name = os.path.split(model_path.rstrip('/'))[-1]+args.mask_encoding_way
    # if args.mask:
    #     if args.RAG:
    #         answers_json_path = f"result/answers{model_name}_mask_RAG_{args.model_iter}.json"
    #     else:
    #         answers_json_path = f"result/answers{model_name}_mask_{args.model_iter}.json"
    # else:
    #     answers_json_path = f"result/answers{model_name}_{args.model_iter}.json"
    answers_json_path = f"result/getpic.json"
    if not os.path.exists("result"):
        os.makedirs("result")

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []
    existing_images = [a["image"] for a in all_answers_json]

    for image_path in chat_ad.keys():
        text_gt = chat_ad[image_path]
        rel_image_path = os.path.join(cfg["data_path"], image_path)
        if image_path in existing_images and not args.reproduce:
            continue
        if image_path=="VisA/pcb1/test/bad/043.JPG":
            print("skip")
            continue
        if args.mask:
            if "DS-MVTec" in image_path:
                expert = 'vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1'
            elif "VisA" in image_path:
                expert = 'vitill_visa_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1'
            elif "GoodsAD" in image_path:
                expert = 'vitill_goods_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1'
            elif "LOCO" in image_path:
                expert = 'vitill_LOCO_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1'
            else:
                raise ValueError("no match expert")
            # rel_mask_path = locate(rel_image_path,args.model_iter,expert)
            rel_mask_path = locate(rel_image_path,500,expert)
        else:
            rel_mask_path = ""
        if args.RAG:
            knowledge = get_combined_and_good(cfg["knowledge_path"],image_path.split('/')[1])
        else:
            knowledge = ""
        qwenquery = QwenQuery(image_path=rel_image_path,mask_path = rel_mask_path, text_gt=text_gt,mask = args.mask,RAG = args.RAG,knowledge=knowledge)
        questions, answers, gpt_answers = qwenquery.generate_answer()
        if gpt_answers is None or len(gpt_answers) != len(answers):
            print(f"Error at {image_path}")
            continue
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        #print(f"Accuracy: {accuracy:.2f}")

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
        # 更新答案记录

        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }

            all_answers_json.append(answer_entry)

        # 保存答案为JSON
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)
    
    caculate_accuracy_mmad(answers_json_path)
