import json
import setting
import pika
from models.VideoCaptionModel import VideoCaptionModel
import requests
from io import BytesIO
import tensorflow as tf

url = "https://api.weixin.qq.com/cgi-bin/media/get?access_token={}&media_id={}"
with tf.device("cpu:0"):
    video_model = VideoCaptionModel()


def video_callback(channel_, method, properties, body):
    header = {"__TypeId__": "answer"}
    user_dict = json.loads(body.decode())
    print(user_dict)
    access_token = user_dict.get("access_token")
    media_id = user_dict.get("media_id")
    video = requests.get(url.format(access_token, media_id)).content
    # print(len(video))
    # print(len(video))
    # print(video[:20])
    video = BytesIO(video)
    answer = video_model.predict(video, url.format(access_token, media_id))
    to_user = dict()
    to_user['answer'] = "{}".format(answer)
    to_user['toUser'] = user_dict.get("toUser")
    print(to_user)
    properties_1 = pika.BasicProperties(content_type="json", delivery_mode=2, headers=header, priority=0)
    channel_.basic_publish(exchange=setting.exchange_name,
                           routing_key='answer.abc', body=json.dumps(to_user), properties=properties_1)
