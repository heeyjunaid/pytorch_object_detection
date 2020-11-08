import torch
from prepare_dataset import PascalVocDataset
from prepare_train import get_model, get_transform
from engine import train_one_epoch, evaluate
from ped_dataset import PennFudanDataset
import utils



def main(root, num_classes, num_epochs, batch_size, data = "r", backbone = None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    
    #dataset = PascalVocDataset(root, get_transform(train=True), data)
    #dataset_test = PascalVocDataset(root, get_transform(train=False), data)

    if data == "ped":
        dataset = PennFudanDataset(root, get_transform(train=True))
        dataset_test = PennFudanDataset(root, get_transform(train=False))
    else:
        dataset = PascalVocDataset(root, get_transform(train=True), data = data)
        dataset_test = PascalVocDataset(root, get_transform(train=False), data= data )

    print("preparing dataset....")
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    print("Done.")

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True
                                            ,collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                            collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes, backbone)

    print(model)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    #load model to cpu
    model = model.to(torch.device("cpu"))
    #save model
    torch.save(model, "./scrap_model.pth")
    print("That's it!")


if __name__ == "__main__":

    #root = "E:/BE Project/code/tomato_data_preprocessing/tomato_img_5mp/"
    #root = "E:/BE Project/Recycle_data/data"
    #root = "E:/BE Project/Fruits Data/train_zip/fruits_data/"
    #root = "D:/Datasets/PennFudanPed"
    
    #root = "D:/Datasets/Scrap"
    root = "E:/Projects/2020/FreeLancing/Raedd/Explainer/Resources/dataset/Dataset/final_localization_data"
    main(root, 2, 10, 1, data = "photoshop")