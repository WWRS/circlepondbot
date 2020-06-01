from PIL import Image
from urllib.request import urlopen
import numpy as np
import tensorflow as tf
from tensorflow import keras
import discord
import os
from io import BytesIO

img_bytes = BytesIO()

# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, 60, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.load_weights("fountain_on.h5")

def get_prediction():
    u = urlopen("https://www.washington.edu/cambots/camera1_l.jpg")
    img = Image.open(u)
    img.save(img_bytes, "JPEG")
    img_bytes.seek(0)
    img = img.crop((419, 250, 449, 310))
    o = model.predict(np.reshape(img,[1,30,60,3]))
    return o[0][1]

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
            title="**I think Drumheller Fountain is {} ({})**".format(prediction, predict_percent),
            color=discord.Colour.from_rgb(51, 0, 111)
        )
        await message.channel.send(file=discord.File(fp=img_bytes,filename="circlepond.jpg"),embed=e)
        #await message.channel.send(embed=e)

@client.event
async def on_ready():
    activity = discord.Activity(name='!circlepond', type=discord.ActivityType.watching)
    await client.change_presence(activity=activity)

client.run(TOKEN)
