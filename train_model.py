import torch
from dataset import PascalVocDataset, PennFudanDataset, CustomDataset
from prepare_train import get_model, get_transform
from engine import train_one_epoch, evaluate
import utils
import os



def main(root, num_classes, num_epochs, batch_size, label_mapping_dict ={}, backbone = None, save_model_path = "./trained_model.pth", save_checkpoints = False, ignore_list=[], data="custom"):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    
    #dataset = PascalVocDataset(root, get_transform(train=True), data)
    #dataset_test = PascalVocDataset(root, get_transform(train=False), data)

    if data == "ped":
        dataset = PennFudanDataset(root, get_transform(train=True))
        dataset_test = PennFudanDataset(root, get_transform(train=False))
    elif data == "custom":
        dataset = CustomDataset(root, get_transform(train=True), label_mapping_dict, ignore_list)
        dataset_test = CustomDataset(root, get_transform(train=False),label_mapping_dict, ignore_list)
    else:
        dataset = PascalVocDataset(root, get_transform(train=True), label_mapping_dict)
        dataset_test = PascalVocDataset(root, get_transform(train=False), label_mapping_dict)

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

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
        if save_checkpoints:
            #load model to cpu
            save_model = model.to(torch.device("cpu"))
            training_checkpoint_path = f"./checkpoints/training_checkpoint_{epoch}/{num_epochs}.pth"
            #save model
            torch.save(save_model, training_checkpoint_path)

        
    #load model to cpu
    model = model.to(torch.device("cpu"))
    #save model
    torch.save(model, save_model_path)
    print("That's it!")


if __name__ == "__main__":

    root = "E:/02 Neurithm/03 BOSH Hackathon/03 Code/01 FasterRCNN/pytorch_object_detection/other_utils/clean_data"
    label_mapping = {'chair':1, 'door':2, 'sofa':3, 'table':4, 'bed':5, 'cupboard':6, 'stool':7}
    ignore_list = ["cupboard", "stool", "bed"]

    main(root, 6, 1, 1, label_mapping, ignore_list=ignore_list)