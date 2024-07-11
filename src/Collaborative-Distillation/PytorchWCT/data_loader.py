import torch.utils.data

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, content_tensor, style_tensor):
        self.content_tensor = content_tensor
        self.style_tensor = style_tensor

    def __len__(self):
        return 1  # Only one pair of images

    def __getitem__(self, idx):
        return self.content_tensor.squeeze(0), self.style_tensor.squeeze(0)
