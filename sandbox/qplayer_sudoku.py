#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# adapted from https://github.com/locuslab/optnet/blob/master/sudoku/train.py

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from proxsuite.torch.qplayer import QPFunction
from torch.nn.parameter import Parameter

N_EPOCH = 2
TRAIN_BATCH_SIZE = 150
TEST_BATCH_SIZE = 200
SUDOKU_MATRIX_SHAPE = (40, 64)  # from data


class QPLayer(nn.Module):
    def __init__(self, n, max_iter=1000):
        super().__init__()
        self.max_iter = max_iter
        nx = (n**2) ** 3
        self.Q = torch.zeros(nx, nx, dtype=torch.float64)
        self.G = -torch.eye(nx, dtype=torch.float64)
        self.u = torch.zeros(nx, dtype=torch.float64)
        self.l = -1.0e20 * torch.ones(nx, dtype=torch.float64)
        self.A = Parameter(
            torch.rand(SUDOKU_MATRIX_SHAPE, dtype=torch.float64)
        )
        self.log_z0 = Parameter(torch.zeros(nx, dtype=torch.float64))

    def forward(self, puzzles):
        n_batch = puzzles.size(0)
        p = -puzzles.view(n_batch, -1)
        b = self.A.mv(self.log_z0.exp())
        x, _, _ = QPFunction(maxIter=self.max_iter)(
            self.Q, p.double(), self.A, b, self.G, self.l, self.u
        )
        return x.float().view_as(puzzles)


def compute_error(pred):
    batch_size = pred.size(0)
    n_sq = int(pred.size(1))
    n = int(np.sqrt(n_sq))
    s = (n_sq - 1) * n_sq // 2  # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batch_size, n_sq, n_sq)

    def invalid_groups(x):
        valid = x.min(1)[0] == 0
        valid *= x.max(1)[0] == n_sq - 1
        valid *= x.sum(1) == s
        return ~valid

    board_correct = torch.ones(batch_size).type_as(pred)
    for j in range(n_sq):
        # Check the jth row and column.
        board_correct[invalid_groups(I[:, j, :])] = 0
        board_correct[invalid_groups(I[:, :, j])] = 0

        # Check the jth block.
        row, col = n * (j // n), n * (j % n)
        M = invalid_groups(
            I[:, row : row + n, col : col + n]
            .contiguous()
            .view(batch_size, -1)
        )
        board_correct[M] = 0

        if board_correct.sum() == 0:
            return batch_size

    return batch_size - board_correct.sum().item()


def train(epoch, model, train_x, train_y, optimizer):
    batch_size = TRAIN_BATCH_SIZE
    batch_data = torch.empty(
        (batch_size, train_x.size(1), train_x.size(2), train_x.size(3)),
        dtype=torch.float32,
    )
    batch_targets = torch.empty(
        (batch_size, train_y.size(1), train_x.size(2), train_x.size(3)),
        dtype=torch.float32,
    )

    for i in range(0, train_x.size(0), batch_size):
        start = time.time()
        batch_data.data[:] = train_x[i : i + batch_size]
        batch_targets.data[:] = train_y[i : i + batch_size]

        optimizer.zero_grad()
        preds = model(batch_data)
        loss = nn.MSELoss()(preds, batch_targets)
        loss.backward()
        optimizer.step()
        err = compute_error(preds.data) / batch_size
        perc = float(i + batch_size) / train_x.size(0) * 100
        print(
            f"Epoch: {epoch} "
            f"[{i + batch_size}/{train_x.size(0)} ({perc:.0f}%)]\t"
            f"Loss: {loss.item():.4f} "
            f"Err: {err:.4f} "
            f"Time: {time.time()-start:.2f} s"
        )


def test(epoch, model, test_x, test_y):
    batch_size = TEST_BATCH_SIZE
    test_loss = 0
    batch_data = torch.empty(
        (batch_size, test_x.size(1), test_x.size(2), test_x.size(3)),
        dtype=torch.float32,
    )
    batch_targets = torch.empty(
        (batch_size, test_y.size(1), test_x.size(2), test_x.size(3)),
        dtype=torch.float32,
    )

    n_err: int = 0
    for i in range(0, test_x.size(0), batch_size):
        print("Testing model: {}/{}".format(i, test_x.size(0)), end="\r")
        with torch.no_grad():
            batch_data.data[:] = test_x[i : i + batch_size]
            batch_targets.data[:] = test_y[i : i + batch_size]
            output = model(batch_data)
            test_loss += nn.MSELoss()(output, batch_targets)
            n_err += compute_error(output.data)

    nBatches = test_x.size(0) / batch_size
    test_loss = test_loss.item() / nBatches
    test_err = n_err / test_x.size(0)
    print("TEST SET RESULTS:" + " " * 20)
    print(f"Average loss: {test_loss:.4f}")
    print(f"Err: {test_err:.4f}")


if __name__ == "__main__":
    # load dataset created with
    # https://github.com/locuslab/optnet/blob/master/sudoku/create.py default
    # board size is 2
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{cur_dir}/data/features.pt", "rb") as f:
        X = torch.load(f)
    with open(f"{cur_dir}/data/labels.pt", "rb") as f:
        Y = torch.load(f)

    N, n_features = X.size(0), int(np.prod(X.size()[1:]))
    n_train = int(0.9 * N)
    n_test = N - n_train
    train_x = X[:n_train]
    train_y = Y[:n_train]
    test_x = X[n_train:]
    test_y = Y[n_train:]
    assert n_train % TRAIN_BATCH_SIZE == 0
    assert n_test % TEST_BATCH_SIZE == 0

    model = QPLayer(n=2)
    lr = 5.0e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    test(0, model, test_x, test_y)
    for epoch in range(1, N_EPOCH + 1):
        print(f"EPOCH {epoch}")
        train(epoch, model, train_x, train_y, optimizer)
        test(epoch, model, test_x, test_y)
