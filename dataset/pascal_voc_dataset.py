import os
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import torch.utils.data



#FIXME: Add support for coco dataset.

class PascalVocDataset(torch.utils.data.Dataset):
    '''
    This function is used to generate a Dataset class for the PascalVoc format dataset.

    Input:
        1. root(str):          root dir/ dataset dir.
        2. transforms(torchvision.transforms): transforms list to be applied on given image.

    Output:
        1.img(torch.tensor):    tensor containg transformed image.
        2.target(dict)     :    dictionary containg bnd boxes and labels.

    Note: target format is like this{'boxes': [N, 4], labels: [N]}
    '''
    def __init__(self, root, transforms, label_mapping_dict = {}):
        self.root = root
        self.transforms = transforms

        #get all anotation files
        self.xml_file = [x for x in os.listdir(root) if x.endswith(".xml")] 

        #label emcoder
        self.label_encoder = label_mapping_dict
        
        
    def __getitem__(self, idx):
        '''
            This function returns image and targets for for file at given index
        '''

        xml_name = self.xml_file[idx]
        xml_tree = ET.parse(os.path.join(self.root, xml_name))
        xml_root = xml_tree.getroot()

        img_path = os.path.join(self.root, xml_root[1].text)
        img = Image.open(img_path).convert("RGB")               #load img in rgb format

        #get all objects
        objects = [child for child in xml_root if child.tag == "object"]

        label_list = []
        bnd_box_list = []

        #iterate over all objects
        for object_ in objects:
            xmin = int(object_[4][0].text)
            ymin = int(object_[4][1].text)
            xmax = int(object_[4][2].text)
            ymax = int(object_[4][3].text)

            #append to list
            label_list.append(self.label_encoder[object_[0].text])      #get encoded label name
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
        return len(self.xml_file)


if __name__ == "__main__":

    root = "E:/Projects/2020/FreeLancing/Raedd/Explainer/Resources/dataset/Dataset/final_localization_data"
    #root = "E:/BE Project/Recycle_data/data"
    dataset = PascalVocDataset(root,  None, data = "photoshop")

    img, target = dataset[0]
    print(np.shape(img))
    print(target)