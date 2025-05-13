# import json
# import os
#
# # 基础路径
# base_path = "/data/guyf/1/diff/datasets/fill50k"
# output_file = "output.jsonl"
#
# # 打开文件准备写入
# with open(output_file, 'w') as f:
#     # 遍历moire文件夹下的所有class
#     for class_dir in os.listdir(os.path.join(base_path, "train_bfhdmi/moire")):
#         if not class_dir.startswith('class4'):
#             continue
#
#         class_path = os.path.join("moire", class_dir)
#         full_class_path = os.path.join(base_path, class_path)
#
#         # 遍历该class下的所有图片
#         for img_name in os.listdir(full_class_path):
#             if not img_name.endswith('.png'):
#                 continue
#
#             # 构建对应的clear图像路径
#             clear_img_path = os.path.join("train_bfhdmi/clear", class_dir, img_name)
#             moire_img_path = os.path.join("train_bfhdmi/moire", class_dir, img_name)
#
#             # 确保clear图像存在
#             if not os.path.exists(os.path.join(base_path, clear_img_path)):
#                 continue
#
#             # 创建JSON对象
#             data = {
#                 "text": "demoire",
#                 "image": moire_img_path,
#                 "conditioning_image": clear_img_path
#             }
#
#             # 写入文件
#             f.write(json.dumps(data) + '\n')
#
# print(f"JSONL文件已生成: {output_file}")

import json


def modify_paths(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)

            # 修改image路径
            if 'image' in data:
                data['image'] = data['image'].replace('train_bfhdmi/source/', 'moire/')

            # 修改conditioning_image路径
            if 'conditioning_image' in data:
                data['conditioning_image'] = data['conditioning_image'].replace('train_bfhdmi/target/', 'clear/')

            # 写入修改后的数据
            outfile.write(json.dumps(data) + '\n')


# 使用示例
modify_paths(r'D:\study\dataset\demoire\train_test.jsonl', r'D:\study\dataset\demoire\output.jsonl')