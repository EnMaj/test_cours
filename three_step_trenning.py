import os

import torch

from dataset_preprocessing_utils import features

from pytorch_training import train_epoch, eval_model
from training_aux import EarlyStopping
from two_step_modeling import CreditsRNN, embedding_projections, device

TRAIN_BUCKETS_PATH = "data/train_buckets_rnn"
VAL_BUCKETS_PATH = "data/val_buckets_rnn"
TEST_BUCKETS_PATH = "data/test_buckets_rnn"

dataset_train = sorted([os.path.join(TRAIN_BUCKETS_PATH, x) for x in os.listdir(TRAIN_BUCKETS_PATH)])
dataset_val = sorted([os.path.join(VAL_BUCKETS_PATH, x) for x in os.listdir(VAL_BUCKETS_PATH)])
dataset_test = sorted([os.path.join(TEST_BUCKETS_PATH, x) for x in os.listdir(TEST_BUCKETS_PATH)])

path_to_checkpoints = "checkpoints/pytorch_baseline/"
es = EarlyStopping(patience=3, mode="max", verbose=True, save_path=os.path.join(path_to_checkpoints, "best_checkpoint.pt"),
                   metric_name="ROC-AUC", save_format="torch")

num_epochs = 2
train_batch_size = 128
val_batch_size = 128

model = CreditsRNN(features, embedding_projections).to(device)
optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    train_epoch(model, optimizer, dataset_train, batch_size=train_batch_size,
                shuffle=True, print_loss_every_n_batches=500, device=device)

    val_roc_auc = eval_model(model, dataset_val, batch_size=val_batch_size, device=device)
    es(val_roc_auc, model)

    if es.early_stop:
        print("Early stopping reached. Stop training...")
        break
    torch.save(model.state_dict(), os.path.join(path_to_checkpoints, f"epoch_{epoch + 1}_val_{val_roc_auc:.3f}.pt"))

    train_roc_auc = eval_model(model, dataset_train, batch_size=val_batch_size, device=device)
    print(f"Epoch {epoch + 1} completed. Train ROC AUC: {train_roc_auc}, val ROC AUC: {val_roc_auc}")

