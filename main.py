import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

mp_selfie = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
with mp_selfie.SelfieSegmentation(model_selection=0) as model:
    if cap.isOpened() == False:
        print("Error in opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret:
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            res = model.process(frame)
            frame.flags.writeable = True

            # Display the resulting frame
            cv2.imshow('Frame',frame)
            # Press esc to exit
            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# plt.figure(figsize=(12,12))
# grid = gridspec.GridSpec(1,2)

# ax0 = plt.subplot(grid[0])
# ax1 = plt.subplot(grid[1])

# ax0.imshow(frame)
# ax1.imshow(res.segmentation_mask)
# plt.show()

mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5
segmented = np.where(mask, frame, cv2.blur(frame, (60,60)))
# background = np.zeros(frame.shape, dtype=np.uint8)
# segmented = np.where(mask, frame, background)
plt.imshow(segmented)
plt.show()