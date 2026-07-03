from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'r') as f:
            magic_num = int.from_bytes(f.read(4), 'big')
            image_count = int.from_bytes(f.read(4), 'big')
            row_count = int.from_bytes(f.read(4), 'big')
            column_count = int.from_bytes(f.read(4), 'big')

            img_data = f.read()
            X = np.frombuffer(img_data, dtype=np.uint8).reshape(image_count, row_count*column_count)

        with gzip.open(label_filename, 'r') as f:
            magic_num = int.from_bytes(f.read(4), 'big')
            label_count = int.from_bytes(f.read(4), 'big')

            label_data = f.read()
            y = np.frombuffer(label_data, dtype=np.uint8).reshape(label_count)

        X = X.astype(np.float32) / 255.0

        self.row_count = row_count
        self.column_count = column_count
        self.transforms = transforms
        self.images = X
        self.labels = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.images[index]

        if isinstance(index, (int, np.integer)):
            X = X.reshape(self.row_count, self.column_count, 1)
        else:
            X = X.reshape(-1, self.row_count, self.column_count, 1)
        if self.transforms is not None:
            if isinstance(index, (int, np.integer)):
                for transform in self.transforms:
                    X = transform(X)
            else:
                X = np.stack([self.apply_transforms(x) for x in X], axis=0)

        return X, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
