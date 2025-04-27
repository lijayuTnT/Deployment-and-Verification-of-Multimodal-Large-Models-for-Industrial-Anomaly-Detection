import gradio as gr
import cv2
import base64
import uuid
import json
from openai import OpenAI
from PIL import Image
from Dinomaly.locate_anomaly import locate

# ============== 你的基础定义保持不变 ==============
instruction = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the input image and answer the questions about it.
When you receive an image,you will be provided with domain konwledge which describe the good case and bad case of the product in the image, you may also be provided with a heatmap highlighting potentially anomalous areas in orange-red. You may use the domain knowledge and the  heatmap to assist in answering questions, but do not mention them, as the user is unaware of its existence.
'''
expert_instruction = '''
You are an image inspector. The categories you can identify include:'breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors','cigarette_box', 'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package',
'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper','candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'.If the object does not fall into any of these categories, classify it as "other".
'''

LOCO = ['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']
Goods = ['cigarette_box','drink_bottle','drink_can','food_bottle','food_box','food_package']
Mvtec = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
Visa = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

client = OpenAI(
    api_key = "sk-3f31b82e72ff4b9897c939d0e37f4f1a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
knowledge_path="../../../dataset/MMAD/domain_knowledge.json"
max_image_size = (512, 512)

# ============== 重要的工具函数 ==============

def encode_image_to_base64(image):
    height, width = image.shape[:2]
    scale = min(max_image_size[0] / width, max_image_size[1] / height)
    new_width, new_height = int(width * scale), int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    _, encoded_image = cv2.imencode('.jpg', resized_image)
    return base64.b64encode(encoded_image).decode('utf-8')

def get_combined_and_good(file_path, item_category):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if item_category not in data.keys():
        raise ValueError(f"类别 '{item_category}' 不存在于数据中。")
    item_data = data[item_category]
    combined = item_data.get("combined", "")
    good = item_data.get("good", "")
    return combined + "\n" + good

def prepare_image_info(image_path):
    image = cv2.imread(image_path)
    base64_image = encode_image_to_base64(image)
    expert_msg = [
        {"role": "system", "content": [{"type": "text", "text": expert_instruction}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Following is the image, when you respond, simply provide the category."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
        ]}
    ]
    # 修改这里：保证传给API的是UTF-8正确编码
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=json.loads(json.dumps(expert_msg, ensure_ascii=False))
    )
    category = completion.choices[0].message.content
    knowledge = get_combined_and_good(knowledge_path, category)

    if category in Mvtec:
        expert = 'vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1'
    elif category in Visa:
        expert = 'vitill_visa_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1'
    elif category in Goods:
        expert = 'vitill_goods_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1'
    elif category in LOCO:
        expert = 'vitill_LOCO_uni_dinov2br_c392r_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa_ghmp09f01w01_b16_ev_s1'
    else:
        expert = None

    mask_path = locate(image_path, 500, expert) if expert else None
    return mask_path, knowledge

def build_chat_messages(image_path, mask_path, knowledge, question):
    image = cv2.imread(image_path)
    base64_image = encode_image_to_base64(image)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Following is the query image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}},
            {"type": "text", "text": "Following is the domain knowledge corresponding to the image."},
            {"type": "text", "text": knowledge}
        ]}
    ]
    if mask_path:
        heat_map = cv2.imread(mask_path)
        base64_heat_map = encode_image_to_base64(heat_map)
        messages[1]["content"].append({"type": "text", "text": "Following is a heatmap corresponding to the image."})
        messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_heat_map}"}})
    messages[1]["content"].append({"type": "text", "text": "Following are the questions:"})
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return messages

# ============== 状态变量管理 ==============
state = {
    "image_uploaded": False,
    "image_path": None,
    "mask_path": None,
    "knowledge": None,
}

# ============== 处理函数 ==============

def handle_upload(image):
    if image:
        unique_filename = f"./chat_pic/temp_{uuid.uuid4().hex}.jpg"
        image.save(unique_filename)
        mask_path, knowledge = prepare_image_info(unique_filename)
        state["image_uploaded"] = True
        state["image_path"] = unique_filename
        state["mask_path"] = mask_path
        state["knowledge"] = knowledge
        return mask_path, knowledge
    return None, None

def handle_question(image, question, chatbox):
    if not state["image_uploaded"]:
        return chatbox + [["系统", "请先上传图片。"]]
    if not question:
        return chatbox + [["系统", "请输入你的问题。"]]
    conversation = build_chat_messages(state["image_path"], state["mask_path"], state["knowledge"], question)
    completion = client.chat.completions.create(model="qwen-vl-max-latest", messages=conversation)
    answer = completion.choices[0].message.content
    chatbox.append(["👤", question])
    chatbox.append(["🤖", answer])
    return chatbox

def quick_task(task, chatbox):
    tasks = {
        "异常检测": "这张图片是否存在异常？",
        "缺陷定位": "异常位于哪里？",
        "缺陷描述": "描述异常情况。",
        "物品分类": "物品是什么？"
    }
    question = tasks.get(task, "")
    return handle_question(None, question, chatbox)

# ============== 界面搭建 ==============

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🚀 Anomaly Agent Plus\n#### 上传图片，智能检测，缺陷分析。")

    chatbot = gr.Chatbot(label="对话记录", height=400, render_markdown=True)

    # 图片 + 热图 + 知识 输出区
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="上传图片", tool="editor", height=280)
        with gr.Column(scale=2):
            mask_output = gr.Image(label="缺陷热图", height=280, width=280)
        with gr.Column(scale=3):
            knowledge_output = gr.Textbox(label="领域知识", lines=10, interactive=False)

    # 上传按钮单独一行，居中美化
    with gr.Row():
        gr.Button("上传并提取", elem_id="extract-btn", size="lg", scale=3).click(
            fn=handle_upload,
            inputs=[image_input],
            outputs=[mask_output, knowledge_output]
        )

    gr.Markdown("---")

    # 提问框 + 提交按钮
    with gr.Row():
        question_input = gr.Textbox(placeholder="输入问题...", label=None, scale=6)
        submit_btn = gr.Button("发送", variant="primary", scale=1)

    submit_btn.click(fn=handle_question, inputs=[image_input, question_input, chatbot], outputs=chatbot)

    # 快捷任务按钮
    gr.Markdown("### 快捷任务按钮：")
    with gr.Row():
        gr.Button("异常检测").click(fn=lambda chatbox: quick_task("异常检测", chatbox), inputs=chatbot, outputs=chatbot)
        gr.Button("缺陷定位").click(fn=lambda chatbox: quick_task("缺陷定位", chatbox), inputs=chatbot, outputs=chatbot)
        gr.Button("缺陷描述").click(fn=lambda chatbox: quick_task("缺陷描述", chatbox), inputs=chatbot, outputs=chatbot)
        gr.Button("物品分类").click(fn=lambda chatbox: quick_task("物品分类", chatbox), inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()