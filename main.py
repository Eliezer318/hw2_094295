import preprocess
import augmentation
import visualize


preprocess.renew_data_files()
preprocess.create_folders()
preprocess.split_train_val(data_path="data", n_images=39)
visualize.stats("data", True)

# augmentation
augmentation.mirror_and_flip(data_path="data", phase="train")
augmentation.image_corners(data_path="data", k=3)
augmentation.blur_augmentation(data_path="data")
augmentation.augment_combinations(data_path="data", phase="train")
visualize.stats("data", True)

from run_train_eval import *
dict_wrongs = visualize.vis(model_ft, val_dataloader)
print(dict_wrongs)
