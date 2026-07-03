from typing import Any, Iterable, Iterator, List, Optional, Sized, Union

import numpy as np

from ..autograd import Tensor


class Dataset:
    """An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """

    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_index = 0

        if self.shuffle is True:
            indices = np.random.permutation(len(self.dataset))
            self.ordering = np.array_split(
                indices, range(self.batch_size, len(self.dataset), self.batch_size)
            )

        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_index >= len(self.ordering):
            raise StopIteration

        curr_batch = self.ordering[self.batch_index]

        self.batch_index += 1
        samples = [self.dataset[idx] for idx in curr_batch]

        if isinstance(samples[0], tuple):
            batch_components = zip(*samples)
            return tuple(
                Tensor(np.stack(component, axis=0)) for component in batch_components
            )
        else:
            return Tensor(np.stack(samples, axis=0))
        ### END YOUR SOLUTION
