import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
            nn.Residual(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    norm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(hidden_dim, dim),
                    norm(dim)
                )
            ),
            nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
            nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    loss_fn = nn.SoftmaxLoss()
    total_loss = 0
    total_samples = 0
    errors = 0

    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)

        if opt is not None:
            loss.backward()
            opt.step()

        total_loss += loss.numpy() * X.shape[0]

        pred = logits.numpy().argmax(axis=1)
        errors += np.sum(pred != y.numpy())
        total_samples += X.shape[0]

    return (errors / total_samples), (total_loss / total_samples)

    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz"
    )

    mnist_train_dataloader = ndl.data.DataLoader(
        dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True
    )

    model = MLPResNet(
        784,
        hidden_dim,
    )

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
        train_error, train_loss = epoch(mnist_train_dataloader, model, opt)

    mnist_test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )

    mnist_test_dataloader = ndl.data.DataLoader(
        dataset=mnist_test_dataset, batch_size=batch_size
    )

    test_error, test_loss = epoch(mnist_test_dataloader, model)

    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
