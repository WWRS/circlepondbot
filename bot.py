from PIL import Image
from urllib.request import urlopen
import numpy as np
import tensorflow as tf
from tensorflow import keras
import discord
import os
from io import BytesIO

img_file = None

# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(60, 30, 3)),
    keras.layers.Dense(2700, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.load_weights("fountain_on.h5")
#print(keras.backend.image_data_format())

def get_prediction():
    global img_file
    u = urlopen("https://www.washington.edu/cambots/camera1_l.jpg")
    img = Image.open(u)
    
    img_bytes = BytesIO()
    img.save(img_bytes, "JPEG")
    img_bytes.seek(0)
    img_file = discord.File(fp=img_bytes,filename="circlepond.jpg")
    
    img = img.crop((421, 249, 451, 309))
    img_arr = np.reshape(img, (1,60,30,3))
    img_arr = np.multiply(img_arr, 1. / 255)
    o = model.predict(img_arr)
    #print(o)
    return o[0][0]

# bot stuff
TOKEN = os.environ.get("TOKEN")
client = discord.Client()

@client.event
async def on_message(message):
    if message.author is client.user:
        return
    if message.content.startswith("!circlepond"):
        predict_prob = get_prediction()
        prediction = "ON" if predict_prob > 0.5 else "OFF"
        predict_percent = "{0:.2%}".format(predict_prob if predict_prob > 0.5 else 1-predict_prob)
        e = discord.Embed(
            title="**I think Drumheller Fountain is {} ({} sure)**".format(prediction, predict_percent),
            color=discord.Colour.from_rgb(51, 0, 111)
        )
        await message.channel.send(file=img_file,embed=e)

@client.event
async def on_ready():
    activity = discord.Activity(name='!circlepond', type=discord.ActivityType.watching)
    await client.change_presence(activity=activity)

client.run(TOKEN)
