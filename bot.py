import urllib.request
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
import discord
import discord.ext

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
    u = urllib.request.urlopen("https://www.washington.edu/cambots/camera1_l.jpg")
    img = Image.open(u)
    
    img_bytes = BytesIO()
    img.save(img_bytes, "JPEG")
    img_bytes.seek(0)
    img_file = discord.File(fp=img_bytes, filename="circlepond.jpg")
    
    img = img.crop((421, 249, 451, 309))
    img_arr = np.reshape(img, (1, 60, 30, 3))
    img_arr = np.multiply(img_arr, 1. / 255)
    o = model.predict(img_arr)
    #print(o)
    return o[0][0]

# bot stuff
TOKEN = os.environ.get("TOKEN")
bot = discord.Client(intents=discord.Intents.default())
tree = discord.app_commands.CommandTree(bot)

def get_embed():
    predict_prob = get_prediction()
    prediction = "ON" if predict_prob > 0.5 else "OFF"
    predict_percent = "{0:.2%}".format(predict_prob if predict_prob > 0.5 else 1-predict_prob)
    
    return discord.Embed(
        title="**I think Drumheller Fountain is {} ({} sure)**".format(prediction, predict_percent),
        color=discord.Colour.from_rgb(51, 0, 111)
    )

@bot.event
async def on_ready():
    await tree.sync()
    activity = discord.Activity(name='/circlepond', type=discord.ActivityType.watching)
    await bot.change_presence(activity=activity)
    print("Running :)")

@tree.command(
    name="circlepond",
    description="Figure out if Drumheller Fountain is on")
async def circlepond(ctx):
    e = get_embed()
    await ctx.response.send_message(file=img_file, embed=e)

bot.run(TOKEN)
