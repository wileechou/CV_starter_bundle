import numpy as np
import cv2

labels = ["dog","cat","pandas"]
np.random.seed(1)

W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread("dog.984.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print("[INFO] {}:{:.2f}".format(label, score))

# draw the label with the highest score on the iamge as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
cv2.imshow("Image", orig)
cv2.waitKey(0)
