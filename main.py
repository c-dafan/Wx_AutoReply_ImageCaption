import pika
import setting
# from callback.ImageCallback import image_callback
from callback.ImageCaptionCallback import image_caption_callback
from callback.VideoCallback import video_callback

credentials = pika.PlainCredentials(setting.username, setting.password)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=setting.host,
                                                               port=setting.port,
                                                               virtual_host="/",
                                                               credentials=credentials
                                                               ))

channel = connection.channel(channel_number=2)
channel.basic_consume(consumer_callback=image_caption_callback, queue=setting.queue_name_img, no_ack=True)
channel.basic_consume(consumer_callback=video_callback, queue=setting.queue_name_video, no_ack=True)
channel.start_consuming()
#
#