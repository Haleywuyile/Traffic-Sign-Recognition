import ttkbootstrap as ttkbootstrap
from keras.preprocessing.image import ImageDataGenerator
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.applications.mobilenet_v2 import  preprocess_input

from keras.models import load_model

model = load_model('TSR_model.h5')
classes={
0:"Speed limit 5",
1:"Speed limit 15",
2:"Speed limit 30",
3:"Speed limit 40",
4:"Speed limit 40",
5:"Speed limit 60",
6:"Speed limit 70",
7:"Speed limit 80",
8:"No Straight and Left Turn",
9:"No Straight and Right Turn",
10:"No Straight Through",
11: "No Left Turn",
12:"No Left and Right Turn",
13:"No Right Turn",
14: "No Overtaking",
15: "No u-turn",
16:"No Motor Vehicle",
17: "Horn Prohibited",
18: "unlimit speed 40",
19: "unlimiti speed 50",
20:"Compulsory Ahead or Turn Right",
21:"Compulsory Ahead",
22:"Compulsory Turn Left",
23:"Compulsory Turn Left and Right",
24:"Compulsory Turn Right",
25:"Drive on Left",
26:"Drive on Right",
27:"Round About",
28:"Car Crossing",
29:"Compulsory Sound Horn",
30:"Cycle Crossing",
31:"U-turn",
32:"Detour",
33:"Traffic Lights Ahead",
34:"Watch for danger",
35:"Watch for Pedestrians",
36:"Watch for non-motor vehicle",
37:"Watch for children",
38:"Right Hand Curve",
39:"Left Hand Curve",
40:"Down the steep slope marks",
41:"Sign on a steep slope",
42:"Slow moving",
43:"Side Road Right",
44:"Side Road Left",
45:"Village Area",
46:"Reverse curve sign",
47:"Unguarded Level Crossing",
48:"Men at work",
49:"winding road",
50:"Guarded Level Crossing",
51:"Accident Area",
52:"NO waiting",
53:"NO entry",
54:"Do not park",
55:"Do not enter",
56:"Yield",
57:"Parking inspection"
}


def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((root.winfo_width() / 2.25), (root.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    # upload image
    sign_image = ttkbootstrap.Label()
    sign_image.pack()
    sign_image.configure(image=im)
    sign_image.image = im
    label.configure(text=' ')
    # after upload the image show the classify button
    show_classify_button(file_path)


def show_classify_button(file_path):
    classify_btn = ttkbootstrap.Button(text="Classify Image", command=lambda: classify(file_path))
    classify_btn.place(x=50, y=10)
    classify_btn.pack(padx=10, pady=5)


def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((224, 224))
    im = np.array(image)
    im = np.expand_dims(im, axis=0)
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    im = datagen.flow(x=im, batch_size=1)
    pred = model.predict(im)
    pred_label = np.argmax(pred[0])
    sign=classes[pred_label]
    print(sign)
    label.configure(text=sign)


# set window
root = ttkbootstrap.Window(
    title="Traffic Sign Recognition",
    themename="superhero",
    size=(1066, 600),
    position=(100, 100),
)
# set heading
heading = ttkbootstrap.Label(text="Traffic Sign Recognition", font=('arial', 20, 'bold'))
heading.pack(pady=20)

# create upload button
ttkbootstrap.Button(root, text="Upload Image", command=upload_image).pack(side=BOTTOM, padx=5, pady=10)

# prediect class
label = ttkbootstrap.Label(font=('arial', 15, 'bold'))
label.pack(side=BOTTOM, expand=True)

root.mainloop()
