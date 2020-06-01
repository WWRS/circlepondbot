from PIL import Image
from urllib.request import urlopen
import numpy as np
import tensorflow as tf
from tensorflow import keras
import discord
import os

os.makedirs("img/img", exist_ok=True)

img = None

# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, 60, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.load_weights('fountain_on.h5')

# generators
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

flow = datagen.flow_from_directory(
    'img',
    target_size=(30,60),
    shuffle=False,
    class_mode='sparse',
    batch_size=1
)

def get_prediction():
    u = urlopen("https://www.washington.edu/cambots/camera1_l.jpg")
    img = Image.open(u)
    img.save("img.jpg")
    img = img.crop((419, 250, 449, 310))
    img.save("img/img/img.jpg")
    o = model.predict_generator(flow)
    return o[0][1]

# bot stuff
TOKEN = os.environ.get('TOKEN',3)
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
        await message.channel.send(file=discord.File("img.jpg"),embed=e)
        #await message.channel.send(embed=e)

@client.event
async def on_ready():
    activity = discord.Activity(name='!circlepond', type=discord.ActivityType.watching)
    await client.change_presence(activity=activity)

client.run(TOKEN)
