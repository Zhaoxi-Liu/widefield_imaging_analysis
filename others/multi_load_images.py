from os.path import join as pjoin
import os
from tifffile import imread
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor


def multi_load_images(image_path_ls, n_thread=20):
    def divide_list(lst, n):
        """将列表等分为n份"""
        size = len(lst) // n
        remainder = len(lst) % n
        chunk_ls = []
        start = 0

        for i in range(n):
            if i < remainder:
                end = start + size + 1
            else:
                end = start + size
            chunk_ls.append(lst[start:end])
            start = end

        return chunk_ls

    def empty_chunk_list(chunk_ls, row=512, col=512):
        empty_chunk_ls = []
        for idx in range(len(chunk_ls)):
            image_chunk = np.empty([len(chunk_ls[idx]), row, col], dtype='uint16')
            empty_chunk_ls.append(image_chunk)
        return empty_chunk_ls

    def load_images2chunk(imagename_ls, image_chunk):
        for idx in range(len(imagename_ls)):
            filename = imagename_ls[idx]
            image = imread(filename)
            image_chunk[idx, :, :] = image

    imagename_chunks = divide_list(image_path_ls, n_thread)
    chunk_ls = empty_chunk_list(imagename_chunks)

    with ThreadPoolExecutor() as executor:
        futureLs = []
        for thread_idx in range(n_thread):
            future = executor.submit(load_images2chunk, imagename_chunks[thread_idx], chunk_ls[thread_idx])
            futureLs.append(future)
        for future in futureLs:
            result = future.result()

    return np.concatenate(chunk_ls)
