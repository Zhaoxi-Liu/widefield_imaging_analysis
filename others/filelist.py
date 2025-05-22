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



#%%
import pandas as pd
import numpy as np

file_path = 'D:\\Desktop\\results_20241231.xlsx'
# df = pd.read_csv(file_path)
df = pd.read_excel(file_path)


# 1. 提取时间部分和序号部分
df[['time_part', 'trial_num']] = df['trials'].str.rsplit('-', n=1, expand=True)
df['trial_num'] = df['trial_num'].astype(int)  # 转换为整数

# 2. 获取唯一时间列表（按原始顺序）
all_times = df['time_part'].unique()  # 自动保留原始顺序

# 3. 对每个时间段的 70 个 trials 进行补全
complete_dfs = []
for time in all_times:
    time_df = df[df['time_part'] == time].copy()

    # 创建完整的 0-69 序列
    full_range = pd.DataFrame({'trial_num': range(70)})

    # 合并（保留原有数据）
    merged = pd.merge(full_range, time_df, on='trial_num', how='left')
    merged['time_part'] = time  # 统一填充当前时间段

    # 重建 trials 名称（缺失的填充为 time-序号）
    merged['trials'] = merged.apply(
        lambda row: f"{time}-{row['trial_num']}"
        if pd.isna(row.get('trials', np.nan))
        else row['trials'],
        axis=1
    )
    complete_dfs.append(merged)

# 4. 合并所有时间段的数据
final_df = pd.concat(complete_dfs)

# 5. 删除 'time_part' 和 'trial_num' 两列，并重置索引
final_df = final_df.drop(columns=['trial_num'])
final_df = final_df.reset_index(drop=True)  # drop=True 避免旧索引成为新列

#%%
import pandas as pd
from os.path import join as pjoin
movie_folder = pjoin(r'Y:\WF_VC_liuzhaoxi\24.12.19_P41\natural-movie\raw', 'natural_movies')
movie_list = pd.read_csv(pjoin(movie_folder, 'movie_list.txt'), header=None).values
n_movie = movie_list.size
movie_name_list = []
for imovie in range(n_movie):
    movie_name_list.append(str(movie_list[imovie])[2:-6])

#%%
stim_list = [item for __ in range(len(all_times)) for item in movie_name_list for _ in range(5)]
# reverse_ls = ['natural-movie' for _ in range(70)] + ['natural-movie-reverse' for _ in range(70)]

#%%
# # 检查行数是否足够，不够则补充空行
# if len(df) < 140:
#     df = df.reindex(range(140), fill_value='')

# (3) 将数据写入第3列
# final_df['video_status'] = reverse_ls
final_df['movie'] = stim_list

#%%
# (4) 保存回CSV
final_df.to_csv('D:\\Desktop\\results_20241231.csv', index=False)

print("数据写入成功！")

#%%
df2=pd.read_excel('D:\\Desktop\\results_20250303.xlsx')
df3 = pd.concat([final_df,df2], axis=0, ignore_index=True)
df3.to_csv('D:\\Desktop\\results_all.csv', index=False)
#%%
import pandas as pd
import matplotlib.pyplot as plt

# 筛选符合条件的行
sucrose5_data = df3[df3['results'] == 'Sucrose5']

# 统计 movie 列的频次
sucrose_counts = sucrose5_data['movie'].value_counts().reset_index()
sucrose_counts.columns = ['movie', 'count']  # 重命名列
print("Sucrose Results Movie Counts:")
print(sucrose_counts)

#%%
plt.figure(figsize=(10, 6))
plt.bar(sucrose_counts['movie'], sucrose_counts['count'])

# 添加图表标签
plt.title('Frequency of Movies for Sucrose5 Results')
plt.xlabel('Movie')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签避免重叠

# 显示数值标签
for i, count in enumerate(sucrose_counts['count']):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('D:\\Desktop\\Frequency of Movies for Sucrose5 Results.png')
plt.show()


#%%
import pandas as pd
import matplotlib.pyplot as plt

# 筛选Water结果的行
water_data = df3[df3['results'] == 'Water']

# 统计movie列的频次
water_counts = water_data['movie'].value_counts().reset_index()
water_counts.columns = ['movie', 'count']  # 重命名列

print("Water Results Movie Counts:")
print(water_counts)

#%%
plt.figure(figsize=(10, 6))
bars = plt.bar(water_counts['movie'], water_counts['count'], color='g')

# 添加图表标签
plt.title('Frequency of Movies for Water Results', pad=20)
plt.xlabel('Movie', labelpad=10)
plt.ylabel('Count', labelpad=10)
plt.xticks(rotation=45, ha='right')  # 右对齐旋转标签

# 显示数值标签
for i, count in enumerate(water_counts['count']):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('D:\\Desktop\\Frequency of Movies for Water Results.png', dpi=300, bbox_inches='tight')
plt.show()

all_times = df['time_part'].unique()