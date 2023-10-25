import lightning.pytorch as pl
from datasets import SegDataModule
from models import SegModel


def main():
    datamodule = SegDataModule(batch_size=8)
    model = SegModel()

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=datamodule)

    
if __name__ == "__main__":
    main()