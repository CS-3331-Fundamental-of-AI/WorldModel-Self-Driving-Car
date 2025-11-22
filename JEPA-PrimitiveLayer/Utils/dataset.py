from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mask import masking, apply_mask

class MapDataset(Dataset):

    def __init__(self, map_csv_file: str):

        self.map_files = pd.read_csv(map_csv_file)
        self.root_dir = "/content/dataset/exported_maps/maps/"

    def __len__(self):
        return len(self.map_files)

    def __getitem__(self, idx, visualize=False):
        # return the masking of that image file
        map_file = self.map_files.iloc[idx, 0] # get the file_name
        full_file_name = self.root_dir + map_file

        mask_emp_np, mask_non_emp_np, mask_union_np, ph, pw, bev, img = masking(full_file_name, visualize)
        mask_emp, mask_non_emp, mask_union = apply_mask(bev, mask_emp_np, mask_non_emp_np, mask_union_np, False)
        return bev, mask_emp, mask_non_emp, mask_union, mask_emp_np, mask_non_emp_np, mask_union_np, ph, pw, img