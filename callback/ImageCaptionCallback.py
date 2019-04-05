import json
import setting
import pika
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from models.ImageCaptionModel import ImageCaptionModel

with tf.device("cpu:0"):
    img_model = ImageCaptionModel()

def image_caption_callback(channel_, method, properties, body):
    header = {"__TypeId__": "answer"}
    user_dict = json.loads(body.decode())
    print(user_dict)
    down_url = user_dict.get("downUrl")
    img = requests.get(down_url).content
    img = BytesIO(img)
    answer = img_model.predict(img)
    to_user = dict()
    to_user['answer'] = "{}".format(answer)
    to_user['toUser'] = user_dict.get("toUser")
    print(to_user)
    properties_1 = pika.BasicProperties(content_type="json", delivery_mode=2, headers=header, priority=0)
    channel_.basic_publish(exchange=setting.exchange_name,
                           routing_key='answer.abc', body=json.dumps(to_user), properties=properties_1)
