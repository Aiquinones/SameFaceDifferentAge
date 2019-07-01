# %%
import cv2
import pickle
import os
import tqdm
import pylab
import numpy as np
from scipy import fftpack
from PIL import Image, ImageEnhance

training_path = "dataset/training"
testing_path = "dataset/testing"


# %%
def get_images_from_filepaths(filepaths):
    imgs = []
    pbar = tqdm.tqdm(total=len(filepaths))
    for filepath in filepaths:
        pbar.update()
        imgs.append(cv2.imread(os.path.expanduser(filepath)))
    pbar.close()

    return imgs


def filter_faces(paths, image_size=160):
    imgs = []

    pbar = tqdm.tqdm(total=len(paths))
    for filepath in paths:
        pbar.update()

        if filepath.split(".")[-2][-1] == "B":
            cropped = cv2.imread(os.path.expanduser(filepath))  # Read
            cropped = cv2.resize(cropped, (image_size, image_size))  # Resize
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # Gray (one channel)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # Gray (three channels)
            imgs.append(gray)
            continue

        im = np.mean(pylab.imread(filepath), axis=2)
        im_fft = fftpack.fft2(im)

        keep_fraction = 0.085
        im_fft2 = im_fft.copy()
        r, c = im_fft.shape
        im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
        im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0

        im_new = fftpack.ifft2(im_fft2).real

        imgsharpimg = Image.fromarray(np.uint8(im_new))
        enhancer = ImageEnhance.Contrast(imgsharpimg)
        enhanced_im = enhancer.enhance(1.8)
        imgcontrast = np.asarray(enhanced_im)

        img = cv2.resize(imgcontrast, (image_size, image_size))
        imgs.append(img)

    pbar.close()
    return imgs, paths


# %%
def save_images(images, folder, filenames, ignored=[]):
    pbar = tqdm.tqdm(total=len(filenames))
    for image, filename in zip(images, filenames):
        pbar.update()

        if filename in ignored:
            continue

        name = filename.split("/")[-1].split(".")[0] + ".png"

        cv2.imwrite(f"{folder}/{name}", image)
    pbar.close()


# %%
if __name__ == "__main__":

    training_path_destination = "dataset/filtered/training"
    testing_path_destination = "dataset/filtered/testing"

    print("Procesando testing...")
    testing_paths = [testing_path + "/" + path for path in os.listdir(testing_path) if path.split(".")[1] == "jpg"]

    removed_testing = []
    cropped_testing, cropped_paths_testing = filter_faces(testing_paths)

    print(removed_testing)

    # %%
    save_images(cropped_testing, testing_path_destination, cropped_paths_testing)

    # %%

    print("Testing:")
    if len(removed_testing) == 0:
        print(f"Todas las caras fueron procesadas! ({len(cropped_testing)}/{len(testing_paths)})")
    else:
        print(f"Ciertas caras ({len(removed_testing)}) no pudieron ser definidas y "
              f"fueron pobladas con un None.\n"
              f"El resto ({len(cropped_testing)}/{len(testing_paths)}) fue procesado correctamente")

    print("Procesando training...")
    training_paths = [training_path + "/" + path for path in os.listdir(training_path) if path.split(".")[1] == "jpg"]
    removed_training = []
    cropped_training, cropped_paths_training = filter_faces(training_paths)

    print(removed_training)

    print("Training:")
    if len(removed_testing) == 0:
        print(f"Todas las caras fueron procesadas! ({len(cropped_training)}/{len(training_paths)})")
    else:
        print(f"Ciertas caras ({len(removed_training)}) no pudieron ser definidas y "
              f"fueron pobladas con un None.\n"
              f"El resto ({len(cropped_training)}/{len(training_paths)}) fue procesado correctamente")

    save_images(cropped_training, training_path_destination, cropped_paths_training)
