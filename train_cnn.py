import os.path
import cv2
import torch.optim
from datasets import CIFARDataset, AnimalDataset
from SimpleCNN import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import shutil

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="/mnt/e/Dataset/animals_v2", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data pre processing
    train_transform = Compose([
            RandomAffine(
                degrees=(-5, 5),
                translate=(0.15, 0.15), #dich chuyen
                scale=(0.85, 1.15),
                shear=5
            ),
            ColorJitter(
                brightness= 0.125,
                contrast= 0.5,
                saturation= 0.25,
                hue= 0.05
            ),

            # thuong dung 1 thay doi vi tri 2 la de thay doi mau sac
            Resize((args.image_size, args.image_size)),
            ToTensor(),
        ])
    test_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
    ])
    train_dataset = AnimalDataset(root=args.root, train=True, transform=train_transform)

    image, _ = train_dataset.__getitem__(200)
    image = torch.permute(image, (1, 2, 0))*255.
    # đặc điểm của pil hay opencv đều cần có gia trị tu 0 255
    # dang dung pytorch co gia tri tu 0 1 nen dua vao cv2 thì p nhan voi 255
    image = image.numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # cv2.imshow("test image", image)
    # # transpose chi co 2 chieu thoi
    # #print(image.shape)
    # cv2.waitKey(0)
    # exit(0)
    # visual thử để xem radom affient - sửa ảnh

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = AnimalDataset(root=args.root, train=False, transform=test_transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.logging)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    num_iters = len(train_dataloader)


    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)   # predictions shape 64x10
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=test_dataset.categories, epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
        # print(classification_report(all_labels, all_predictions))




