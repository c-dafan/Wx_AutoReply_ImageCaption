import json
import setting
import pika
from keras.applications.nasnet import NASNetMobile
from keras.applications.xception import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
model = NASNetMobile()


def image_callback(channel_, method, properties, body):
    header = {"__TypeId__": "answer"}
    user_dict = json.loads(body.decode())
    print(user_dict)
    down_url = user_dict.get("downUrl")
    img = requests.get(down_url).content
    img = BytesIO(img)
    img = image.load_img(img, target_size=(224, 224))
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(img_arr)
    preds = model.predict(x)
    answer = decode_predictions(preds, top=3)[0]
    to_user = dict()
    to_user['answer'] = "{}".format(answer)
    to_user['toUser'] = user_dict.get("toUser")
    print(to_user)
    properties_1 = pika.BasicProperties(content_type="json", delivery_mode=2, headers=header, priority=0)
    channel_.basic_publish(exchange=setting.exchange_name,
                           routing_key='answer.abc', body=json.dumps(to_user), properties=properties_1)
