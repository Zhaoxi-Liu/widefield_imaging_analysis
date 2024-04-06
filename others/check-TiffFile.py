from tifffile import TiffFile, imread, imwrite

f = r'Y:\WF_VC_liuzhaoxi\24.01.03_C83\retinotopy\raw\20240103-123936-405.tif'
tmp = TiffFile(f)
dims = [len(tmp.pages), *tmp.pages[0].shape]
print(dims)
# image = imread(f)
# print(image.shape)
# imwrite(f, image)
# imwrite(f+'1', image, imagej=True)


