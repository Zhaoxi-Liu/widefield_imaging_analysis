import os

# 获取用户输入的目标文件夹路径
target_path = input("请输入目标文件夹的路径: ")+"raw/"

# 检查目标路径是否存在
if not os.path.exists(target_path):
    print("目标文件夹路径不存在，请重新输入有效路径。")
else:
    # 获取目标文件夹下的所有文件和文件夹
    items = os.listdir(target_path)

    # 筛选出文件夹并添加双引号
    folders = ['"{}"'.format(item) for item in items if os.path.isdir(os.path.join(target_path, item))]

    # 定义要保存的txt文件名
    txt_file = target_path+'../datalist.txt'

    # 将文件夹名称用逗号分隔并保存到txt文件
    with open(txt_file, 'w') as file:
        folder_names = ', '.join(folders)
        file.write(folder_names)

    print(f"数据列表已用逗号分隔并加上双引号保存到 {txt_file} 文件中。")
