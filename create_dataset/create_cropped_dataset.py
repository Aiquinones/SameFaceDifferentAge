#%%
import cv2
import pickle
import os
import tqdm

#%%
def get_images_from_filepaths(filepaths):
    imgs = []
    pbar = tqdm.tqdm(total=len(filepaths))
    for filepath in filepaths:
        pbar.update()
        imgs.append(cv2.imread(os.path.expanduser(filepath)))
    pbar.close()
        
    return imgs

def crop_faces(paths, removed=[], image_size=160):
    # Images already crop, resize
    imgs = []
    pbar = tqdm.tqdm(total=len(paths))
    for filepath in paths:
        pbar.update()
        cropped = cv2.imread(os.path.expanduser(filepath))
        cropped = cv2.resize(cropped, (image_size, image_size))
        imgs.append(cropped)
    pbar.close()
    return imgs, paths
        
#%%
training_path = "dataset/training"
testing_path = "dataset/testing"

training_path_destination = "dataset/cropped/training"
testing_path_destination = "dataset/cropped/testing"

#%%
def save_images(images, folder, filenames, ignored=[]):
    pbar = tqdm.tqdm(total=len(filenames))
    for image, filename in zip(images, filenames):
        pbar.update()
        
        if filename in ignored:
            continue
        
        name = filename.split("/")[-1].split(".")[0] + ".png"
        
        if "226930" in name:
            print(name)

        cv2.imwrite(f"{folder}/{name}", image)
    pbar.close()


#%%
print("Procesando testing...")
testing_paths = [testing_path + "/" + path for path in os.listdir(testing_path) if path.split(".")[1] == "jpg"]
removed_testing = []
cropped_testing, cropped_paths_testing = crop_faces(testing_paths, removed_testing)

print(removed_testing)

#%%
save_images(cropped_testing, testing_path_destination, cropped_paths_testing)

#%%

print("Testing:")
if len(removed_testing) == 0:
    print(f"Todas las caras fueron procesadas! ({len(cropped_testing)}/{len(testing_paths)})")
else:
    print(f"Ciertas caras ({len(removed_testing)}) no pudieron ser definidas y fueron pobladas con un None.\nEl resto ({len(cropped_testing)}/{len(testing_paths)}) fue procesado correctamente")

print("Procesando training...")
training_paths = [training_path + "/" + path for path in os.listdir(training_path) if path.split(".")[1] == "jpg"]
removed_training = []
cropped_training, cropped_paths_training = crop_faces(training_paths, removed_training)

print(removed_training)


print("Training:")
if len(removed_testing) == 0:
    print(f"Todas las caras fueron procesadas! ({len(cropped_training)}/{len(training_paths)})")
else:
    print(f"Ciertas caras ({len(removed_training)}) no pudieron ser definidas y fueron pobladas con un None.\nEl resto ({len(cropped_training)}/{len(training_paths)}) fue procesado correctamente")

save_images(cropped_training, training_path_destination, cropped_paths_training)