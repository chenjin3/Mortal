import json

def json_to_jsonl(json_file_path, jsonl_file_path):
    # 打开JSON文件并加载数据
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # 打开JSONL文件以写入数据
    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for item in data:
            # 将每个字典对象转换为JSON字符串并写入文件，每个对象占一行
            jsonl_file.write(json.dumps(item) + '\n')

# 使用示例
json_to_jsonl('test_aliyun.json', 'quehun.jsonl')