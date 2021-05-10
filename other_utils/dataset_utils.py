import os
import shutil
import json


def has_required_classes(objects, allowed_classes):

    has_classes = []

    for object_ in objects:
        has_classes.append(object_["name"] in allowed_classes)

    return any(has_classes)




def clean_data(root_dir, allowed_classes = [], copy_dir = "./clean_data"):

    if not os.path.exists(copy_dir):
        os.makedirs(copy_dir)

    json_files = [x for x in os.listdir(root) if x.endswith(".json")]

    for json_file in json_files:
        file_path = os.path.join(root, json_file)

        with open(file_path) as f:
            json_ = json.load(f)

        annotation = json_["annotation"]

        if has_required_classes(annotation["objects"], allowed_classes):
            img_name = annotation["filename"]
            img_path = os.path.join(root_dir, img_name)
            
            shutil.move(img_path, os.path.join(copy_dir, img_name))
            shutil.move(file_path, os.path.join(copy_dir, json_file))



if __name__ == "__main__":
    root = "E:/02 Neurithm/03 BOSH Hackathon/02 Data/04 MIT Indoor scene/formated_data"

    clean_data(root, ['chair', 'door', 'sofa', 'table'])
