import json

def convert_format(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换后的数据列表
    converted_data = []
    
    # 遍历每个样本进行转换
    for item in data:
        converted_item = {
            "messages": [
                {
                    "content": f"<image>{item['problem']}",
                    "role": "user"
                },
                {
                    "role": "assistant",
                    "content": f"{item['thinking']}\n\n{item['answer']}"
                }
            ],
            "images": [
                item['image_path']
            ]
        }
        converted_data.append(converted_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = "data_use/merged_correct_samples-450.json"
    output_file = "data_use/merged_correct_samples-450_sharegpt.json"
    convert_format(input_file, output_file)
    print("转换完成！")
