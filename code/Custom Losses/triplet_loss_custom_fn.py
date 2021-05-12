# import necessary modules
import tensorflow as tf
import face_recognition as fr
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# paths of anchor,positive,negative images
anchor_path = os.path.join(os.getcwd(), "anchor.jpg")  # path to anchor image
pos_path = os.path.join(os.getcwd(), "positive.jpg")  # path to positive image
neg_path = os.path.join(os.getcwd(), "negative.jpg")  # path to negative image


# encodings of images
def encodings(img_path):
    img_arr = fr.api.load_image_file(img_path, mode='RGB')
    bnd_box = fr.api.face_locations(img_arr, number_of_times_to_upsample=1, model='hog')
    encoding = fr.api.face_encodings(img_arr, known_face_locations=bnd_box, num_jitters=1, model='small')

    return bnd_box, encoding


# finding distance between images
def image_dist(anchor_path, pos_path, neg_path):
    _, anchor_encoding = encodings(anchor_path)
    _, pos_encoding = encodings(pos_path)
    _, neg_encoding = encodings(neg_path)
    dist_arr = fr.api.face_distance([pos_encoding[0], neg_encoding[0]], anchor_encoding[0])

    return dist_arr


# Final Triplet Loss function
def TripletLoss(anchor_path, pos_path, neg_path, alpha=2.0):
    _, anchor_encoding = encodings(anchor_path)
    _, pos_encoding = encodings(pos_path)
    _, neg_encoding = encodings(neg_path)
    dist_arr = fr.api.face_distance([pos_encoding[0], neg_encoding[0]], anchor_encoding[0])
    loss = max(dist_arr[0] - dist_arr[1] + alpha, 0)
    return loss


loss = TripletLoss(anchor_path, pos_path, neg_path, alpha=2.0)
print("Loss(General Function): {}".format(loss))


# Implementation in TensorFlow
class TripletLossFn(tf.keras.losses.Loss):
    def __init__(self, alpha=2.0, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss = max(y_true - y_pred + self.alpha, 0)
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "alpha": self.alpha}


loss_obj = TripletLossFn(2.0)
dist_arr = image_dist(anchor_path, pos_path, neg_path)
loss = loss_obj(dist_arr[0], dist_arr[1])
print("Loss (implementation in TensorFlow): {}".format(loss))

anchor_arr = fr.api.load_image_file(anchor_path, mode='RGB')
pos_arr = fr.api.load_image_file(pos_path, mode='RGB')
neg_arr = fr.api.load_image_file(neg_path, mode='RGB')

fig = plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(anchor_arr)
plt.title("Anchor Image")
plt.subplot(1,3,2)
plt.imshow(pos_arr)
plt.title("Positive Image")
plt.subplot(1,3,3)
plt.imshow(neg_arr)
plt.title("Negative Image")

plt.show()