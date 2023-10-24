import numpy as np
from os.path import join as pjoin

path = 'Z:\\WF_VC_liuzhaoxi\\23.10.5_D53_gp5.17\\retinotopy_10x4\\process\\20231005-195517-merged'

U = np.load(pjoin(path, 'U.npy'))
height, width, components = U.shape
U_rot90 = np.empty((width, height, components))
for i in range(components):
    U_rot90[:, :, i] = np.rot90(U[:, :, i], k=3)  # 逆时针旋转90度
np.save(pjoin(path, 'U_rot90.npy'), U_rot90)
