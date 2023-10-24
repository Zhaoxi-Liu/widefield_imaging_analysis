from wfield import *
from tifffile import imwrite

# load data
localdisk = 'Z:\\WF_VC_liuzhaoxi\\23.10.16_D50\\retinotopy_10x4\\process\\20231016-162110-merged'
dat_path = glob(pjoin(localdisk,'*.bin'))[0]

dat = mmap_dat(dat_path)
frames_average = np.load(pjoin(localdisk,'frames_average.npy'))
# imwrite(pjoin(localdisk,"motion_correct.tif"), dat, imagej=True)

U,SVT = approximate_svd(dat, frames_average)
np.save(pjoin(localdisk,'U.npy'),U)
np.save(pjoin(localdisk,'SVT.npy'),SVT)