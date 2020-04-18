import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision.io import read_video



class Video_inference():
    def __init__(self, video_path, model_path, label_dict):
        '''
        Class to perform inference on the video files.
        '''
        self.video_file_path = video_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path).to(self.device)
        self.label_dict = label_dict
        self.transform_ = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 640)), transforms.ToTensor()])

    
    def infer_video(self):
        '''
            Function to return video frame tensors.
        '''

        cap = cv2.VideoCapture(self.video_file_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./output.mp4',fourcc, 20.0, (640, 480))

        with torch.no_grad():
            self.model.eval()

            while(cap.isOpened()):

                grabbed , frame = cap.read()
                
                if not grabbed:
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                tensor_frame = self.array_to_tensor(frame_)
                tensor_frame = tensor_frame.unsqueeze(0).to(self.device)
                output = self.model(tensor_frame)
                img = self.draw_bounding_boxes(frame, output)
                out.write(frame)
        #     tensor_cache.append(tensor_frame)

        # tensor_cache = torch.stack(tensor_cache)

        # return tensor_cache            



    def array_to_tensor(self, img_arry):
        '''
            Function to convert given array to tensor.
        '''
        tensor_ = self.transform_(img_arry)

        return tensor_

    def draw_bounding_boxes(self, img, output, delta = 0.70):
        '''
            Function to draw bnding boxes.
        '''
        boxes = output[0]['boxes'].detach().cpu().numpy()
        scores = output[0]['scores'].detach().cpu().numpy()
        labels = output[0]['labels'].detach().cpu().numpy()

        l = (scores > delta).astype(int)
        b = l.reshape(-1, 1)

        #remove boxeses whose score is less than delta. 
        boxes = (b*boxes).astype(int)
        labels = l*labels

        print(labels)
        print(scores)

        color_list = [(0, 0, 255), (0, 128, 255), (0, 255, 0), (255, 0, 0), (255, 255, 51), (0, 255, 255), (255, 0, 255)]

        for box, label, score in zip(boxes, labels, scores):

            if score == 0:
                break

            if label>0:

                color = color_list[label]
                label = self.label_dict[label]
                text = f"{label}"

                img = cv2.rectangle(img, (box[0],  box[1]), (box[2], box[3]), color , 2)
                
                img = cv2.putText(img, text, (box[0],  box[1]), cv2.FONT_HERSHEY_DUPLEX , 1, color, 2)

        return img









if __name__ == "__main__":

    model_path = 'E:/Projects/2020/pytorch_trained_models/scrap/scrap_frcnn50_model_16_4_2019.pth'  #"./example.pth"
    #video_path = 'E:/Projects/2020/object_detection_pytorch/test_data/metal_test.mp4' #'./example.mpg'
    video_path = 'E:/Projects/2020/object_detection_pytorch/test_data/test4.mp4' #'./example.mpg'
    
    label_dict =  {1: 'paper-Recyclable', 2: 'metal-Recyclable', 3: 'plastic-Recyclable', 4:'rubber-Recyclable', 5: 'ceramic-NonRecyclable', 6:'wrapper-NonRecyclable'}

    ins = Video_inference(video_path, model_path, label_dict)

    ins.infer_video()

        