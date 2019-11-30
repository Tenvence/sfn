from torch.utils.data import DataLoader
from utils.gravel_dataset import GravelDataset
import cv2

gravel_dataset = GravelDataset(is_train=True)
data_loader = DataLoader(gravel_dataset, batch_size=4, shuffle=True)


