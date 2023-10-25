import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import SegDataModule
from models import SegModel


def main():
    datamodule = SegDataModule(batch_size=8)
    model = SegModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_high_vegetation_IoU",
        filename="{epoch}-{val_loss:.2f}-{val_high_vegetation_IoU:.2f}-{val_mIoU:.2f}",
        save_top_k=3,
    )

    trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback], log_every_n_steps=10)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
