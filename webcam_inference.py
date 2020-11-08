import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision.io import read_video
import asyncio
import time


#self.video_file_path = video_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_ = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 640)), transforms.ToTensor()])


async def infer_video(model_path, label_dict):
    '''
        Function to return video frame tensors.
    '''

    model = torch.load(model_path).to(device)
    cap = cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('./output.mp4',fourcc, 20.0, (640, 480))

    with torch.no_grad():
        model.eval()
        while True:

            grabbed , frame = cap.read()
            
            if not grabbed:
                break

            frame = cv2.resize(frame, (640, 480))
            frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = await predict(frame_, frame, model, label_dict)

            # Display the resulting frame
            cv2.imshow('Video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows() 

       
async def predict(frame_, frame, model, label_dict):
    '''
        Asynchronus function for bounding box prediction
    '''
    tensor_frame = array_to_tensor(frame_)
    tensor_frame = tensor_frame.unsqueeze(0).to(device)
    output = model(tensor_frame)
    img = draw_bounding_boxes(frame, output, label_dict)
    return img


def array_to_tensor(img_arry):
    '''
        Function to convert given array to tensor.
    '''
    tensor_ = transform_(img_arry)

    return tensor_


def draw_bounding_boxes(img, output, label_dict, delta = 0.70):
    '''
        Function to draw bnding boxes.
    '''
    since = time.time()

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
            label = label_dict[label]
            text = f"{label}"

            img = cv2.rectangle(img, (box[0],  box[1]), (box[2], box[3]), color , 2)
            
            img = cv2.putText(img, text, (box[0],  box[1]), cv2.FONT_HERSHEY_DUPLEX , 1, color, 2)

    print(f"total time elapsed: {time.time() - since}")
    
    return img


if __name__ == "__main__":

    model_path = 'E:/Projects/2020/pytorch_trained_models/scrap/scrap_frcnn50_model_16_4_2019.pth'  #"./example.pth"
    #video_path = 'E:/Projects/2020/object_detection_pytorch/test_data/metal_test.mp4' #'./example.mpg'
    video_path = 'E:/Projects/2020/object_detection_pytorch/test_data/test4.mp4' #'./example.mpg'
    
    label_dict =  {1: 'paper-Recyclable', 2: 'metal-Recyclable', 3: 'plastic-Recyclable', 4:'rubber-Recyclable', 5: 'ceramic-NonRecyclable', 6:'wrapper-NonRecyclable'}

    #run asynchronus function
    asyncio.get_event_loop().run_until_complete(infer_video( model_path, label_dict))

    
        