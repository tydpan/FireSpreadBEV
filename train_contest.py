import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from evaluate import evaluate
from focal_loss import focal_loss
from unet import UNet
from utils.dice_score import dice_loss
from VLS import (
    CenterCrop,
    Compose,
    MaskToLabel,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
    VLSDataset,
    evaluate_acc,
)

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
data_root = "./data"


def train_net(
    net,
    optimizer,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    # val_percent: float = 0.1,
    save_checkpoint: bool = True,
    # img_scale: float = 0.5,
    amp: bool = False,
    dir_checkpoint: Path = Path("./checkpoints/"),
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    train_set = VLSDataset(
        data_root,
        is_train=True,
        transform=Compose(
            (
                ToTensor(),
                Normalize(
                    mean=[0.796, -5.76e-4, 0.358, 0.230, 3.03e-5, 0.116, -16.068, -21.332],
                    std=[2.241, 0.645, 0.888, 8.35e-4, 3.76e-4, 5.84e-2, 509.35, 547.55],
                ),
                # Normalize(  # minmax
                #     mean=[
                #         -1.079e01,
                #         -7.506,
                #         -6.381,
                #         1.173e-01,
                #         -6.415e-04,
                #         1.396e-03,
                #         -9.748e04,
                #         -1.139e05,
                #     ],
                #     std=[
                #         2.589e01,
                #         1.554e01,
                #         1.830e01,
                #         1.127e-01,
                #         8.082e-02,
                #         4.271e-01,
                #         1.878e05,
                #         1.408e05,
                #     ],
                # ),
                # Normalize(
                #     mean=[0.447, 0.483, 0.368, 0.997, 0.008, 0.269, 0.519, 0.809],
                #     std=[0.0865, 0.0415, 0.0485, 0.00741, 0.00465, 0.137, 0.00271, 0.00389],
                # ),
                MaskToLabel(boundaries=[325, 425, 525, 625, 725]),
                RandomCrop(400),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            )
        ),
    )
    val_set = VLSDataset(
        data_root,
        is_train=False,
        transform=Compose(
            (
                ToTensor(),
                Normalize(
                    mean=[0.796, -5.76e-4, 0.358, 0.230, 3.03e-5, 0.116, -16.068, -21.332],
                    std=[2.241, 0.645, 0.888, 8.35e-4, 3.76e-4, 5.84e-2, 509.35, 547.55],
                ),
                # Normalize(  # minmax
                #     mean=[
                #         -1.079e01,
                #         -7.506,
                #         -6.381,
                #         1.173e-01,
                #         -6.415e-04,
                #         1.396e-03,
                #         -9.748e04,
                #         -1.139e05,
                #     ],
                #     std=[
                #         2.589e01,
                #         1.554e01,
                #         1.830e01,
                #         1.127e-01,
                #         8.082e-02,
                #         4.271e-01,
                #         1.878e05,
                #         1.408e05,
                #     ],
                # ),
                # Normalize(
                #     mean=[0.447, 0.483, 0.368, 0.997, 0.008, 0.269, 0.519, 0.809],
                #     std=[0.0865, 0.0415, 0.0485, 0.00741, 0.00465, 0.137, 0.00271, 0.00389],
                # ),
                MaskToLabel(boundaries=[325, 425, 525, 625, 725]),
                CenterCrop(400),
            )
        ),
    )

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(
    #     dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    # )
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            # val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            # img_scale=img_scale,
            amp=amp,
        )
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optimizer(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=60
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = focal_loss(alpha=[0.1, 1, 1, 1, 1, 1], gamma=2, device=device)
    # criterion = focal_loss(gamma=5, device=device)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images = batch["image"]
                true_masks = batch["mask"]

                assert images.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) + dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True,
                    )
                    # loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({"train loss": loss.item(), "step": global_step, "epoch": epoch})
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                # division_step = n_train // (10 * batch_size)
            division_step = 1
            if division_step > 0:
                if global_step % division_step == 0:
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace("/", ".")
                        histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                        histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(net, val_loader, device)
                    accuracy, accuracy_fg = evaluate_acc(net, val_loader, device)
                    # prec, rec, fs = evaluate_acc(net, val_loader, device)
                    # scheduler.step(val_score)
                    scheduler.step(accuracy_fg)
                    # scheduler.step(fs)

                    logging.info("Validation Dice score: {:.4f}".format(val_score))
                    logging.info(
                        f"Validation Dice score: {val_score:.6f}, Acc: {accuracy:.6f}, Acc fg: {accuracy_fg:.6f}"
                    )
                    # logging.info(
                    #     f"Validation Dice score: {val_score:.6f}, precision: {prec:.6f}, recall: {rec:.6f}, fscore: {fs:.6f}"
                    # )
                    experiment.log(
                        {
                            "learning rate": optimizer.param_groups[0]["lr"],
                            "validation acc": accuracy,
                            "validation acc fg": accuracy_fg,
                            # "val precision": prec,
                            # "val recall": rec,
                            # "val fscore": fs,
                            "validation Dice": val_score,
                            # "images": wandb.Image(images[0].cpu()),
                            "masks": {
                                "true": wandb.Image(true_masks[0].float().cpu() / 6),
                                "pred": wandb.Image(
                                    torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
                                    / 6
                                ),
                            },
                            "step": global_step,
                            "epoch": epoch,
                            **histograms,
                        }
                    )

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(
                net.state_dict(), str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch))
            )
            logging.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--optimizer",
        "-o",
        dest="optimizer",
        metavar="OPT",
        type=str,
        default="rms",
        help="Optimizer",
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument(
        "--output-dir", "-d", type=Path, default=Path("./checkpoints/"), help="output directory"
    )
    # parser.add_argument(
    #     "--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images"
    # )
    # parser.add_argument(
    #     "--validation",
    #     "-v",
    #     dest="val",
    #     type=float,
    #     default=10.0,
    #     help="Percent of the data that is used as validation (0-100)",
    # )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument("--classes", "-c", type=int, default=6, help="Number of classes")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=8, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    if args.optimizer == "rms":
        optimizer = optim.RMSprop
    elif args.optimizer == "sgd":
        optimizer = optim.SGD
    else:
        raise

    net.to(device=device)
    try:
        train_net(
            net=net,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            # img_scale=args.scale,
            # val_percent=args.val / 100,
            amp=args.amp,
            dir_checkpoint=args.output_dir,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        raise
