import tensorflow as tf
import logging
import cv2
import numpy as np
import pickle
import imageio
logging.basicConfig(level=logging.INFO)


class VideoCaptionModel(object):
    def __init__(self, feature_len=80):
        logging.info("video_caption 开始创建")
        nasn_model = tf.keras.applications.xception.Xception()
        self.max_len = 25
        self.feature_len = feature_len
        inputs_name = ["a", "b", "c"]
        self.data_root = "resources/"
        self.word_to_id = pickle.load(open(self.data_root + "/video_word/word2id.plk", "rb"))
        self.id_to_word = pickle.load(open(self.data_root + "/video_word/id2word.plk", "rb"))
        self.img_size = 299
        s2vt_model = s2vt_my(input_names=inputs_name, num_word=len(self.word_to_id), len_sentence=self.max_len + 1)
        s2vt_model.summary()
        s2vt_model.load_weights(self.data_root + "/video_caption_weight/")
        self.encode_model = get_encode_model(s2vt_model)
        self.decode_model = get_decode_model(s2vt_model, inputs_name[2])

        self.model = tf.keras.Model(inputs=[nasn_model.input],
                                    outputs=[nasn_model.get_layer("avg_pool").output])
        self.BOS = "<B>"
        self.PAD = "<P>"
        self.EOS = "<E>"
        logging.info("video_caption 加载完成")

    def __get_video(self, file, url):
        logging.info("获取视频")
        # video = cv2.VideoCapture(file)
        # cv2.VideoCapture()
        imgs = []
        try:
            video = imageio.get_reader(file, "ffmpeg")
            for img in video:
                img = cv2.resize(img, (self.img_size, self.img_size))
                imgs.append(img)
        except RuntimeError as e:
            try:
                video = cv2.VideoCapture(file)
                read_true, img = video.read()
                while read_true:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    imgs.append(img)
                    read_true, img = video.read()
                video.release()
            except:
                logging.error("视频无法解析")
        if len(imgs) == 0:
            return np.array(imgs)
        # read_true, img = video.read()
        # while read_true:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     logging.info(img.shape)
        #     img = cv2.resize(img, (self.img_size, self.img_size))
        #     imgs.append(img)
        #     read_true, img = video.read()
        # video.release()
        imgs = np.array(imgs)
        imgs = self.__to_size(imgs)
        imgs = tf.keras.applications.nasnet.preprocess_input(imgs)
        feature = self.model.predict(imgs, batch_size=10)
        return feature

    def __to_size(self, feature):
        if feature.shape[0] < self.feature_len:
            return feature
        feature_size = self.feature_len - 1
        jg = (feature.shape[0] - 1) // feature_size
        res = feature[::jg][-self.feature_len:]
        logging.log(logging.DEBUG, "{} -- {}".format(res.shape, feature.shape))
        return res

    def predict(self, file, url):
        features = self.__get_video(file=file, url=url)
        if features.shape[0] == 0:
            return "抱歉，视频无法解析!!"
        features = np.expand_dims(features, 0)
        word_id = self.word_to_id[self.BOS]
        state_h, state_c = self.encode_model.predict(features)
        logging.debug(state_c.shape)
        logging.debug(state_h.shape)
        sentence = ""
        for word_arg in range(self.max_len + 1):
            predictions, state_h, state_c = self.decode_model.predict(
                [np.array([word_id]).reshape([-1, 1]), state_h, state_c])
            word_id = np.argmax(predictions[0, 0, :])
            word = self.id_to_word[word_id]
            if word == self.EOS:
                break
            sentence += word
            sentence += " "
        return sentence


def s2vt_my(num_word=2200, feature_size=2048, input_names=None, lstm_unit=512, len_sentence=26, embedding_dim=256,
            frame_num=80):
    if input_names is None:
        input_names = [None, None, None]
    feature_input = tf.keras.Input(shape=[frame_num, feature_size], name=input_names[0])
    encode_lstm = tf.keras.layers.LSTM(lstm_unit, return_state=True)
    _, state_h, state_c, = encode_lstm(feature_input)
    decode_input = tf.keras.Input(shape=[len_sentence], name=input_names[1])
    decode_embedding = tf.keras.layers.Embedding(num_word, embedding_dim)(decode_input)
    decode_lstm = tf.keras.layers.LSTM(lstm_unit, return_state=True, return_sequences=True)
    decode_out, _, _ = decode_lstm(decode_embedding, initial_state=[state_h, state_c])
    # out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_word, activation="softmax"), name=input_names[2])(
    #     decode_out)
    out = tf.keras.layers.Dense(num_word, activation="softmax", name=input_names[2])(decode_out)
    model = tf.keras.Model(inputs=[feature_input, decode_input], outputs=[out])
    # model.summary()
    return model


# s2vt_my(input_names=["a", "b", "c"])
def get_encode_model(model_):
    img_feature = tf.keras.Input(shape=[None, 2048])
    lstm = model_.get_layer("lstm_1")
    lstm.trainable = False
    _, h, c = lstm(img_feature)
    return tf.keras.Model(inputs=[img_feature], outputs=[h, c])


def get_decode_model(model_, dense_name):
    state_h = tf.keras.Input(shape=[512])
    state_c = tf.keras.Input(shape=[512])
    text = tf.keras.Input(shape=[1])
    embedding = model_.get_layer("embedding_1")
    embedding.trainable = False
    xx = embedding(text)
    lstm_1 = model_.get_layer("lstm_2")
    lstm_1.trainable = False
    out, d_state_h, d_state_c = lstm_1(xx, initial_state=[state_h, state_c])
    dense = model_.get_layer(dense_name)
    dense.trainable = False
    out = dense(out)
    return tf.keras.Model(inputs=[text, state_h, state_c], outputs=[out, d_state_h, d_state_c])
