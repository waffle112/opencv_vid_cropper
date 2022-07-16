import cv2
import numpy as np
import torchvision
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
import imageio
import imageio_ffmpeg

# lets try and use this
# currently using
# https://github.com/spmallick/learnopencv/blob/master/PyTorch-Keypoint-RCNN/run_pose_estimation.py

# try this next
#can't use - requires nvidia graphics card and cuda
# https://github.com/spmallick/learnopencv/tree/master/Human-Action-Recognition-Using-Detectron2-And-Lstm

#this one looks promising
# https://github.com/ultralytics/yolov3


# create a model object from the keypointrcnn_resnet50_fpn class
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()

# create the list of keypoints.
keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
             'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
             'left_ankle', 'right_ankle']

green = (173, 255, 47)  # yellow/green


def get_file_type(filename):
    return filename[len(filename) - filename[::-1].index('.'):]


def write_file(frames, name, file_type):
    print("Saving " + file_type + " file")
    with imageio.get_writer("videos/out/" + name + "." + file_type, mode="I") as writer:
        for idx, frame in enumerate(frames):
            print("Adding frame to " + file_type + " file: ", idx + 1, end="\r")
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
    ]
    return limbs


limbs = get_limbs_from_keypoints(keypoints)


def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # modify keypoints to also grab the corners?
    boundbox = []

    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
    # iterate for every person detected

    # for person_id in range(len(all_keypoints)):
    for person_id in range(1):
        # check the confidence score of the detected person
        if confs[person_id] > conf_threshold:
            # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]
            # grab the keypoint-scores for the keypoints
            scores = all_scores[person_id, ...]
            # iterate for every keypoint-score
            for kp in range(len(scores)):
                # check the confidence score of detected keypoint
                if scores[kp] > keypoint_threshold:
                    # convert the keypoint float-array to a python-list of intergers
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    # pick the color at the specific color-id
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1]) * 255)
                    # draw a cirle over the keypoint location
                    cv2.circle(img_copy, keypoint, 30, color, -1)
                    boundbox.append(keypoint)

    return img_copy, boundbox


def draw_skeleton_per_person(img, output, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # check if the keypoints are detected
    if len(output["keypoints"]) > 0:
        # pick a set of N color-ids from the spectrum
        colors = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
        # iterate for every person detected
        # for person_id in range(len(all_keypoints)):
        for person_id in range(1):
            # check the confidence score of the detected person
            if confs[person_id] > conf_threshold:
                # grab the keypoint-locations for the detected person
                keypoints = all_keypoints[person_id, ...]

                # iterate for every limb
                for limb_id in range(len(limbs)):
                    # pick the start-point of the limb
                    limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                    # pick the start-point of the limb
                    limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                    # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
                    limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                    # check if limb-score is greater than threshold
                    if limb_score > keypoint_threshold:
                        # pick the color at a specific color-id
                        color = tuple(np.asarray(cmap(colors[person_id])[:-1]) * 255)
                        # draw the line for the limb
                        cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 25)

    return img_copy


def human_recog(file):
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    regFrames = []
    keyFrames = []
    skeFrames = []
    cropFrames = []
    boxFrames = []
    boundboxes = []
    frame_num = 0
    padbox = [0,0,0,0]
    while True:
        print("converting - Frame " + str(frame_num) + "/" + str(totalframes), end='\r')
        frame_num += 1

        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape

        # pre-process image
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(frame)

        # forward-pass the model
        # the input is a list, hence the output will also be a list
        output = model([img_tensor])[0]

        keypoints_img, boundbox = draw_keypoints_per_person(frame, output["keypoints"], output["keypoints_scores"],
                                                            output["scores"], keypoint_threshold=1)
        # skeletal_img = draw_skeleton_per_person(frame, output, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=1)

        regFrames.append(frame)
        keyFrames.append(keypoints_img)
        # skeFrames.append(skeletal_img)

        # print(boundbox)
        #padbox = [0, height, 0, width]
        padbox = [width, 0, height, 0]
        if boundbox:
            x1, x2 = min(_[0] for _ in boundbox), max(_[0] for _ in boundbox)
            y1, y2 = min(_[1] for _ in boundbox), max(_[1] for _ in boundbox)

            # print(temp)
            percentage = .2
            x1 = int(x1 - percentage * (x2 - x1))
            x2 = int(x2 + percentage * (x2 - x1))
            y1 = int(y1 - percentage * (y2 - y1))
            y2 = int(y2 + percentage * (y2 - y1))
            cropFrames.append(frame[max(y1, 0):min(y2, height), max(x1, 0):min(x2, height)])
            boundboxes.append([x1,x2,y1,y2])
            #with each frames positions, x1, x2, y1, y2
            #find the pos that is closest to the borders
            if padbox[0] > x1:
                padbox[0] = x1
            if padbox[1] < x2:
                padbox[1] = x2
            if padbox[2] > y1:
                padbox[2] = y1
            if padbox[3] < y2:
                padbox[3] = y2

            # boxframe = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # boxFrames.append(boxframe)

        # print(skeletal_img, end = "\r")

    print("DONE")

    # outfile = "videos/out/OUT" + file
    # out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'DIVX'), fps, (height, width))
    # for f in range(len(skeFrames)):
    #     out.write(skeFrames[f])
    # out.release()
    padbox = [max(0, padbox[0]), min(width, padbox[1]), max(0, padbox[2]), min(height, padbox[3])]
    # pad croppedframes
    print(padbox)
    px1, px2, py1, py2 = padbox
    for i, f in enumerate(cropFrames):
        #padding is outward
        #height1, width1, _ = f.shape
        x1, x2, y1, y2 = boundboxes[i]

        top = abs(py2 - y2)
        bottom = abs(y1 - py1)
        left = abs(x1 - px1)
        right = abs(px2 - x2)
        #top, bottom, left, right -> y2, y1, x1, x2
        cropFrames[i] = cv2.copyMakeBorder(f, top, bottom, left, right, cv2.BORDER_CONSTANT)

    # write_file(skeFrames, "sketest", get_file_type(file))
    write_file(regFrames, "regtest", get_file_type(file))
    write_file(keyFrames, "keytest", get_file_type(file))
    write_file(cropFrames, "croptest", get_file_type(file))
    # write_file(boxFrames, "boxtest", get_file_type(file))

    cap.release()
    cv2.destroyAllWindows()
    return 1

if __name__ == "__main__":
    file = "videos/dance_test.gif"
    # file = "videos/dance1.mp4"
    # file = "videos/fortnite.mp4"
    human_recog(file)
