import json
import os

def convert_to_infer_format(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 准备输出数据
    output_data = []
    
    for line in lines:
        data = json.loads(line)
        text = data['text']
        label = data['ground_truth']
        image_path = data['image']
        
        # 构建problem
        problem = (
            f"As a hateful speech detection expert, you are presented with a meme sample containing an image and the overlaid text: \"{text}\".\n\n"
            f"Hateful speech is defined as a direct or indirect attack on people based on characteristics such as ethnicity, race, nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, disability, or disease. "
            f"Such attacks may include dehumanizing comparisons (e.g., likening people to animals), statements of inferiority, calls for exclusion or segregation, or mocking hate crimes.\n\n"
            f"Exceptions: Attacks targeting individuals (e.g., public figures) that are not based on protected characteristics, or criticisms directed at hate groups (e.g., terrorist organizations), are **not** considered hateful.\n\n"
            f"Instructions:\n"
            f"1. Carefully analyze the visual and textual content.\n"
            f"2. Consider both overt and subtle hateful expressions based on the definition above.\n"
            f"3. Provide your reasoning process within <think> </think>.\n"
            f"4. Give your final decision using either 'hateful' or 'not-hateful', enclosed in <answer> </answer> tags."
        )
        
        # 构建ground_truth
        gt = "<answer> hateful </answer>" if label == 1 else "<answer> not-hateful </answer>"
        
        # 构建输出格式
        output_item = {
            "prompt": problem,
            "response": gt,
            "images": image_path
        }
        
        output_data.append(output_item)
    
    # 写入输出文件（JSON格式）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 设置输入输出文件路径
    input_file = "data_use/FHM_test_seen_label01.jsonl"
    output_file = "data_use/FHM_test_seen_infer_format.json"
    
    # 执行转换
    convert_to_infer_format(input_file, output_file)
    print(f"转换完成！输出文件保存在: {output_file}")
