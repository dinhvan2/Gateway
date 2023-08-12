from tkinter import *
from keras.models import load_model
from PIL import Image, ImageTk
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


# Resize dimensions for the camera feed
resize_width, resize_height = 224, 224

app = Tk()
app.title("mecanum")
app.geometry("1050x500")
app.configure(bg='black')

mainFrame = Frame(app,height = 300, width = 500)
mainFrame.place(x=258,y=50)
mainFrame.config(highlightbackground='black')

def show_frame():

    if cam_on:

        ret, frame = cap.read()    

        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
            img = Image.fromarray(cv2image).resize((500,300))
            imgtk = ImageTk.PhotoImage(image=img)        
            vid_lbl.imgtk = imgtk    
            vid_lbl.configure(image=imgtk)    
        
        # Preprocess the image
        frame = preprocess_image(frame)

        # Predict the model
        prediction = model.predict(frame)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        ai_message["text"] = "Class:" + class_name[2:] + " Value:" + str(np.round(confidence_score * 100))[:-2] + "%"
        vid_lbl.after(1, show_frame)
        
def start():
    global cam_on, cap
    stop()
    cam_on = True
    cap = cv2.VideoCapture(0)
    show_frame()


def stop():
    global cam_on
    cam_on = False
    
    if cap:
        cap.release()

vid_lbl = Label(mainFrame)
vid_lbl.grid(row=0, column=0)
cam_on = False
cap = None

# Function to preprocess the image
def preprocess_image(image):
    # Resize the raw image
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, resize_width, resize_height, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    return image

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


def try2():
            mqttClient.publish(MQTT_TOPIC_PUB,"tien")
def try3():
            mqttClient.publish(MQTT_TOPIC_PUB,"lui")
def try4():
            mqttClient.publish(MQTT_TOPIC_PUB,"trai")
def try5():
            mqttClient.publish(MQTT_TOPIC_PUB,"phai")
 
# Create a button to open the camera in GUI app
ai_message = Label(app,width= 15,height=2, highlightbackground='black', text="", font=("Arial", 12))
ai_message.place(x=473, y=400)
photo = PhotoImage(file = r"documents/up.png")
image = photo.subsample(8, 8) 
photo2 = PhotoImage(file = r"documents/down.png")
image2 = photo2.subsample(8, 8) 
photo3 = PhotoImage(file = r"documents/left.png")
image3 = photo3.subsample(8, 8) 
photo4 = PhotoImage(file = r"documents/right.png")
image4 = photo4.subsample(8, 8) 

button2 = Button(app, text="Turn on camera", highlightbackground='black', width=25, font=("Arial") , command=start)
button2.place(x=5,y=150)
button3 = Button(app, text='Stop camera',highlightbackground='black', width=25, font=("Arial"), command=stop)
button3.place(x=5,y=175)
button = Button(app, text='Quit', width=25,highlightbackground='black', font=("Arial"), command=app.destroy)
button.place(x=5,y=200)
button4 = Button(app, text='Tien',highlightbackground='black',image = image, command=try2)
button4.place(x=860,y=100)
button5 = Button(app, text='Lui',highlightbackground='black',image = image2, command=try3)
button5.place(x=860,y=180)
button6 = Button(app, text='Trai', highlightbackground='black',image = image3, command=try4)
button6.place(x=780,y=145)
button7 = Button(app, text='Phai', highlightbackground='black',image = image4, command=try5)
button7.place(x=940,y=145)

# Create an infinite loop for displaying app on screen
app.mainloop()


