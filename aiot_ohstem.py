from keras.models import load_model
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time

MQTT_SERVER = "mqtt.ohstem.vn"
MQTT_PORT = 1883
MQTT_USERNAME = "IOT"
MQTT_PASSWORD = ""
MQTT_TOPIC_PUB = MQTT_USERNAME + "/feeds/V10"
MQTT_TOPIC_SUB = MQTT_USERNAME + "/feeds/V11"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
# An IP of your camera can be used as well
camera = cv2.VideoCapture("http://192.168.1.27:81/stream")
# camera = cv2.VideoCapture(0)


# Resize dimensions for the camera feed
resize_width, resize_height = 224, 224

# Function to preprocess the image
def preprocess_image(image):
    # Resize the raw image
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, resize_width, resize_height, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    return image

def ai_detector():
    # Grab the web camera's image.
    ret, image = camera.read()

    # Show the image in a window
    cv2.imshow("Webcam Image", image)
    cv2.waitKey(1)

    # Preprocess the image
    image = preprocess_image(image)

    # Predict the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    return class_name[2:].strip("\n")

def mqtt_connected(client, userdata, flags, rc):
    print("Connected successfully!!")
    client.subscribe(MQTT_TOPIC_SUB)

def mqtt_subscribed(client, userdata, mid, granted_qos):
    print("Subscribed to Topic!!!")

mqttClient = mqtt.Client()
mqttClient.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqttClient.connect(MQTT_SERVER, int(MQTT_PORT), 60)

# Register mqtt events
mqttClient.on_connect = mqtt_connected
mqttClient.on_subscribe = mqtt_subscribed

mqttClient.loop_start()
count = 0
try:
    while True:
        ai_result = ai_detector()
        if ai_result != 'none' :
            count+=1
            if count>=20:
                count = 0
                print(count)
                print(ai_result)
                print(1)
                mqttClient.publish(MQTT_TOPIC_PUB, ai_result)
except KeyboardInterrupt:
    # Release the camera and close OpenCV windows on KeyboardInterrupt (Ctrl+C)
    camera.release()
    cv2.destroyAllWindows()
