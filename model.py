import cv2
from roboflow import Roboflow
from ultralytics import YOLO
from PIL import Image
import pickle


def Download_Dataset():
    rf = Roboflow(api_key="edE8nSa5Ba4DTYwoMsPW")
    project = rf.workspace("vibhu-raj-ysy7d").project("bone_tumor")
    version = project.version(3)
    dataset = version.download("yolov8")


def Train_Model():
    model = YOLO('yolov8n.pt')
    model.train(data='Bone_Tumor-3/data.yaml', epochs=10)
    # paste path properly in data.yaml
    model = YOLO("runs/detect/train/weights/best.pt")

    return model


def Prediction(imgPath: str):
    img = cv2.imread(imgPath)

    # Perform inference (object detection)
    results = model(source=img)

    # Print the results (bounding boxes and class labels)
    for r in results:
        print(r.boxes)

    for i, r in enumerate(results):
        # Save results to disk
        r.save(filename=f'results{i}.jpg')


if __name__ == '__main__':
    # First only run Download Dataset function
    # Download_Dataset()

    # Then inside Bone_Tumor-3 there is a file named data.yaml
    # Open it and copy absolute path of train and valid folder, paste it inside the data.yaml file (see line 11 and 12 in data.yaml) there paste it
    # After updating data.yaml file, uncomment below code and comment Download_Dataset() function
    # Now you can train model and save it
    model = Train_Model()

    # Save the model
    with open('bone_model.pickle', 'wb') as file:
        pickle.dump(model, file)
