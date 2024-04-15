import time
import os


def load_data(folder_name):
    since = time.time()
    
    # Check if folder name is empty or white-space only
    if not folder_name or not folder_name.strip():
        raise ValueError("Folder name must be a string")
    

    folder_path = os.path.abspath(folder_name)

    print('folder path: {}'.format(folder_path))

    subfolder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    print('subfolder names: {}'.format(subfolder_names))

    if len(subfolder_names) <= 1:
        subfolder_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        print('subfolder paths: {}'.format(subfolder_paths))
        subsubfolder_paths = [os.path.join(subfolder_paths[0], name) for name in os.listdir(subfolder_paths[0]) if os.path.isdir(os.path.join(subfolder_paths[0], name))]
        DATA_PATH = subfolder_paths[0]
        NORMAL_PATH = subsubfolder_paths[0]
        COVID_PATH = subsubfolder_paths[1]
        CAP_PATH = subsubfolder_paths[2]
        load_data_corr = 1

    elif len(subfolder_names) > 1:
        subfolder_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        DATA_PATH = folder_path
        NORMAL_PATH = subfolder_paths[0]
        COVID_PATH = subfolder_paths[1]
        CAP_PATH = ""
        load_data_corr = 1
    
    else:
        load_data_corr = 0

    time_since = time.time() - since

    print('-' * 10)
    print('Main path: {}'.format(DATA_PATH))
    print()
    print('Image Path: {} | {} | {}'.format(NORMAL_PATH, COVID_PATH, CAP_PATH))
    print()
    print('Data load completed')
    print('-' * 10)
    print()

    del since, time_since

    return DATA_PATH, NORMAL_PATH, COVID_PATH, CAP_PATH




# class_names = os.listdir(DATA_PATH)
# image_count = {}
# for i in class_names:
#     image_count[i] = len(os.listdir(os.path.join(DATA_PATH,i)))

# #Plotting Distribution of Each Classes
# fig1, ax1 = plt.subplots()
# ax1.pie(image_count.values(),
#         labels = image_count.keys(),
#         shadow=False,
#         autopct = '%1.1f%%',
#         startangle=90)
# plt.show()

# fig = plt.figure(figsize=(16,5))
# fig.suptitle("COVID19 Positive", size=22)
# img_paths = os.listdir(COVID_PATH)
# shuffle(img_paths)

# for i,image in enumerate(img_paths[:4]):
#     img = cv2.imread(os.path.join(COVID_PATH, image))
#     plt.subplot(1,4, i+1, frameon=False)
#     plt.imshow(img)
# fig.show()
# plt.show()

# NORMAL_PATH = "C:/Users/PC/Documents/FYP/FYP/Code/curated_data/train/1NonCOVID"
# COVID_PATH = "C:/Users/PC/Documents/FYP/FYP/Code/curated_data/train/2COVID"
# CAP_PATH = "C:/Users/PC/Documents/FYP/FYP/Code/curated_data/train/3CAP"

# DATA_PATH = "C:/Users/PC/Documents/FYP/FYP/Code/curated_data/train"
