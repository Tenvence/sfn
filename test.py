from torch.utils.data import DataLoader
from utils.gravel_dataset import GravelDataset
import cv2
from PIL import Image
import numpy as np

gravel_dataset = GravelDataset(is_train=True)[0]
