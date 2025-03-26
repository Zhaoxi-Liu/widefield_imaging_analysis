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

#%% 文件重命名
import os
from os.path import join as pjoin

path=r'Y:\WF_VC_liuzhaoxi\P73\20240718\natural-movie\process\20240718-170059-natural-movie\20240718-170059-tif-rep-reshape'
strA = '.tif'
strB = '-rep-reshape.tif'
# 获取当前目录下的所有文件
files = os.listdir(path)

# 遍历文件
for filename in files:
    # 检查文件名中是否包含'10.tif'
    if strA in filename:
        # 创建新文件名
        new_filename = filename.replace(strA, strB)
        # 重命名文件
        os.rename(pjoin(path, filename), pjoin(path, new_filename))
        print(f"Renamed '{filename}' to '{new_filename}'")


#%% 批量复制文件夹
import os
from os.path import join as pjoin
from glob import glob
import shutil

file_ls = glob(pjoin(r'Y:\WF_VC_liuzhaoxi\*\natural-movie-reverse'))
print('\n'.join(file_ls))

#%%
# 目标目录
new_folder = r'Y:\WF_VC_liuzhaoxi\natural-movie data'
# 确保目标目录存在
os.makedirs(new_folder, exist_ok=True)
# 复制文件夹
for folder in file_ls:
    # 获取文件夹名称
    folder_name = os.path.basename(folder)
    # 目标路径
    new_path = pjoin(new_folder, folder_name)
    # 复制文件夹
    shutil.copytree(folder, new_path)

print("复制完成！")

## another example

path_out = 'Z:\MouseWatch'
os.makedirs(path_out, exist_ok=True)

for path_wfield in path_list: 
    print('\n\n'+path_wfield)
    experiment = os.path.basename(path_wfield)[:15]
    mouse = os.path.normpath(path_wfield).split(os.sep)[2].split('_')[1]
    print(experiment, mouse)
    trialfile = pd.read_csv(pjoin(path_wfield, 'trials.csv'), header=None).values.astype(int)
    # filter incomplete experiments
    if trialfile.shape[0] < 70:
        print("This experiment is not complete, skip.")
        continue

    path_event = pjoin(path_wfield, '../../raw/',experiment+'-event')
    # extract stim inf from file path
    parts = path_wfield.split('\\')
    stim = parts[3]
    path_event_new = pjoin(path_out, f'{stim}_{experiment}_face')
    trialfile_new = pjoin(path_out, f'{stim}_{experiment}_trials.csv')
    print(path_event_new, '\n',trialfile_new)
    # copy folder
    shutil.copytree(path_event, path_event_new)
    print(f'finish {path_event_new}')
    # copy file
    shutil.copyfile(pjoin(path_wfield, 'trials.csv'), trialfile_new)
    print(f'finish {trialfile_new}')
    
print("文件复制完成！")
    
