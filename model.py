import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from torchsummary import summary

import os

class RoadsRegionsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Populate image paths and labels
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # print("idx",idx)
        img_path = self.image_paths[idx]
        # print("image path",img_path)
        label = self.labels[idx]
        # print("label",type(label))

        image = Image.open(img_path).convert('L')  # Open as grayscale
        if self.transform:
            image = self.transform(image)

        return image, label

train_dir = r"C:\Users\VAIO\Downloads\Roads_and_regions_data_for_classification_23_jan_2025\Roads_and_regions_data_for_classification_23_jan_2025"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((500, 500)),  # Resize image to 500 * 500
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])
train_dataset = RoadsRegionsDataset(root_dir=train_dir,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
model = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
# print("model fully conneted features",model.fc.in_features)
num_classes = 2
in_features = 2048
model.fc = torch.nn.Linear(in_features, num_classes)
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

summary(model,(3,500,500))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)








