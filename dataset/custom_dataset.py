import os
import json
import torch
import numpy as np
from PIL import Image
import torch.utils.data



class CustomDataset(torch.utils.data.Dataset):
    '''
    This function is used to generate a Dataset class for the PascalVoc format dataset.

    Input:
        1. root(str):          root dir/ dataset dir.
        2. transforms(torchvision.transforms): transforms list to be applied on given image.
        3. label_mapping_dict(dictionary): dictionary mapping for labels

    Output:
        1.img(torch.tensor):    tensor containg transformed image.
        2.target(dict)     :    dictionary containg bnd boxes and labels.

    Note: target format is like this{'boxes': [N, 4], labels: [N]}
    '''
    def __init__(self, root, transforms, label_mapping_dict = {}, ignore_list = []):
        self.root = root
        self.transforms = transforms

        #get all anotation files
        self.json_file = [x for x in os.listdir(root) if x.endswith(".json")] 

        #label emcoder
        self.label_encoder = label_mapping_dict
        self.ignore_list = ignore_list
        
        
    def __getitem__(self, idx):
        '''
            This function returns image and targets for for file at given index
        '''

        json_name = self.json_file[idx]
        json_path = os.path.join(self.root, json_name)

        with open(json_path) as f:
            json_file = json.load(f)

        annotation = json_file["annotation"]

        img_path = os.path.join(self.root, annotation["filename"])
        img = Image.open(img_path).convert("RGB")               #load img in rgb format

        #get all objects
        objects = annotation["objects"]

        label_list = []
        bnd_box_list = []

        #iterate over all objects
        for object_ in objects:
            #don't added bounding boxes of ignored classes.
            if object_["name"] not in self.ignore_list: 
                xmin = object_["bndbox"]["xmin"]
                ymin = object_["bndbox"]["ymin"]
                xmax = object_["bndbox"]["xmax"]
                ymax = object_["bndbox"]["ymax"]

                #append to list
                label_list.append(self.label_encoder[object_["name"]])      #get encoded label name
                bnd_box_list.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(bnd_box_list, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(label_list, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] =  (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(labels)), dtype=torch.int64)
            
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
            
    def __len__(self):
        return len(self.json_file)


if __name__ == "__main__":

    label_mapping = {'chair':1, 'door':2, 'sofa':3, 'table':4, 'bed':5, 'cupboard':6, 'stool':7}
    ignore_list = ["cupboard", "stool", "bed"] #ignore this classes from database

    root = "E:/02 Neurithm/03 BOSH Hackathon/02 Data/04 MIT Indoor scene/formated_data"
    #root = "E:/BE Project/Recycle_data/data"
    dataset = CustomDataset(root,  None, label_mapping, ignore_list)

    img, target = dataset[0]
    print(np.shape(img))
    print(target)
