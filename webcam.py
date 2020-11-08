import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import time


def webcam_inference(MODEL_PATH):
    '''
    This function is used to inference model on webacm stream
    '''
    model = torch.load(MODEL_PATH)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = transforms.ToTensor()

    #move model to device and eval mode
    model.to(device)
    model.eval()
    
    #turn on webcam 
    cap = cv2.VideoCapture(0)

    while(True):

        since = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tensor =  transform(img).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        #predict image
        out = model(img_tensor)

        print(f"prediction from image: {out}")

        print(f"Total time taken: {time.time() - since}")

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    MODEL_PATH = "E:/Projects/2020/object_detection_pytorch/trained_models/pennFun_local_11feb.pth"
    
    webcam_inference(MODEL_PATH)