import tensorflow as tf
import argparse
import os
import numpy as np
import pandas as pd

from model_toad_single_task import TOAD_fc_single_task

class SingleTaskNPZDataset:
    def __init__(self, csv_path, data_dir):
        df = pd.read_csv(csv_path)
        self.slide_ids = df["slide_id"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.data_dir = data_dir

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide = self.slide_ids[idx]
        label = self.labels[idx]

        npz_path = os.path.join(self.data_dir, slide)
        arr = np.load(npz_path)

        # support both: arr["feat"] or arr[arr.files[0]]
        if "feat" in arr:
            feat = arr["feat"]
        else:
            feat = arr[arr.files[0]]

        feat = tf.convert_to_tensor(feat, dtype=tf.float32)
        return feat, label

def get_loader(dataset):
    def gen():
        for i in range(len(dataset)):
            feat, label = dataset[i]
            yield feat, tf.convert_to_tensor([label], dtype=tf.int32)

    output_signature = (
        tf.TensorSpec(shape=(None, 1024), dtype=tf.float32),  # bag
        tf.TensorSpec(shape=(1,), dtype=tf.int32),            # label
    )

    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

def train_one_epoch(loader, model, optimizer, loss_fn):
    epoch_loss = 0.0
    count = 0

    for bag, label in loader:
        with tf.GradientTape() as tape:
            out = model(bag)
            logits = out["logits"]
            loss = loss_fn(label, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss += float(loss.numpy())
        count += 1

    return epoch_loss / max(count, 1)

def main(args):

    dataset = SingleTaskNPZDataset(args.csv, args.data_dir)
    loader = get_loader(dataset)

    model = TOAD_fc_single_task(size_arg="big", dropout=False, n_classes=args.n_classes)
    optimizer = tf.keras.optimizers.Adam(args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(args.epochs):
        loss = train_one_epoch(loader, model, optimizer, loss_fn)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data.csv")
    parser.add_argument("--data_dir", type=str, default="npz_files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()

    main(args)
