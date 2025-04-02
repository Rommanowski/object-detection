import torch
import torchvision
import cv2
import numpy as np
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

coco_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
idx_to_class = {}
for k, v in enumerate(coco_classes):
    idx_to_class[k] = v


def visualize(image, predictions, threshold=0.5):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    ax = plt.gca()
    for idx in range(len(predictions["boxes"])):
        score = predictions["scores"][idx].item()
        if score >= threshold:
            box = predictions["boxes"][idx].cpu().numpy()
            label = predictions["labels"][idx].item()
            label = idx_to_class[label]

            # Draw bounding box
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1], f"{label}, Score: {score:.2f}", color="red")

    plt.axis("off")
    plt.show()


def visualize_one(image, predictions, threshold=0.5):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    ax = plt.gca()
    score = predictions['scores'][0].item()
    box = predictions["boxes"][0].cpu().numpy()
    label = predictions["labels"][0].item()
    label = idx_to_class[label]

    # Draw bounding box
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(box[0], box[1], f"{label}, Score: {score:.2f}", color="red")

    plt.axis("off")
    plt.show()

def get_box(pred):
    box = pred["boxes"][0].cpu().numpy()
    label = pred["labels"][0].item()
    label = idx_to_class[label]

    return box[0], box[1], box[2], box[3], label

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FPS, 2)
if not stream.isOpened():
    print('Camera not found :(')
    exit()

transform = T.ToTensor()
while (True):

    ret, BGR_frame = stream.read()
    frame = cv2.cvtColor(BGR_frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(frame_tensor)

    x1, y1, x2, y2, label = get_box(prediction[0])

    if not ret:
        print('No more stream')
        break

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.rectangle(BGR_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(BGR_frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    isolated_object_frame = BGR_frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.imshow('Isolated object', isolated_object_frame)
    cv2.imshow('Webcam', BGR_frame)

stream.release()
cv2.destroyAllWindows()