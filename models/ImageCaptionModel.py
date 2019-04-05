import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

class ImageCaptionModel(object):
    def __init__(self):
        model = nic(num_word=2200, dim_word=512, lstm_units=512, mode_train=False)
        model.summary()
        logging.info("image_caption 开始创建")
        self.data_root = "resources/"
        model.load_weights(self.data_root + "image_caption_weight/")
        self.img_model = model_decode(model, 224)
        self.decode_model = model_word(model, 512)
        self.img_size = 224
        self.id_to_word = pickle.load(open(self.data_root + "image_word/id_to_word.p", "rb"))
        self.word_to_id = pickle.load(open(self.data_root + "image_word/word_to_id.p", "rb"))
        self.max_taken = 31
        self.BOS = "<B>"
        self.PAD = "<P>"
        self.EOS = "<E>"
        logging.info("image_caption 加载完成")

    def predict(self, file):
        img_inp = self.__get_image(file)
        a, b, c = self.img_model.predict(img_inp)
        word_id = np.argmax(a[0], axis=-1)
        sentence = self.id_to_word[word_id[0]] + " "
        for word_arg in range(self.max_taken):
            a, b, c = self.decode_model.predict([word_id.reshape([-1, 1]), b, c])
            word_id = np.argmax(a[0], axis=-1)
            word = self.id_to_word[word_id[0]]
            if word == self.EOS:
                break
            sentence += word
            sentence += " "
        logging.info(sentence)
        return sentence

    def __get_image(self, file):
        img = tf.keras.preprocessing.image.load_img(file,
                                                    target_size=(self.img_size, self.img_size))
        img = tf.keras.preprocessing.image.img_to_array(img)
        # plt.imshow(img.astype("int"))
        # plt.show()
        img = np.expand_dims(img, 0)
        img = tf.keras.applications.xception.preprocess_input(img)
        return img

def model_decode(model_, img_size):
    img_inp = tf.keras.Input(shape=[img_size, img_size, 3])

    model_img_ = tf.keras.Model(inputs=[model_.input[0]], outputs=[model_.get_layer("lambda").output], name="inp")
    out_, b, c = model_.get_layer("lstm")(model_img_(img_inp))
    out_ = model_.get_layer("time_distributed")(out_)
    return tf.keras.Model(inputs=[img_inp], outputs=[out_, b, c])


def model_word(model_, state_size):
    img_inp = tf.keras.Input(shape=[None])
    state_a = tf.keras.Input(shape=[state_size])
    state_b = tf.keras.Input(shape=[state_size])
    word_embedding = model_.get_layer("embedding")(img_inp)
    lstm = model_.get_layer("lstm")
    out_, a, b = lstm(word_embedding, initial_state=[state_a, state_b])
    out_ = model_.get_layer("time_distributed")(out_)
    return tf.keras.Model(inputs=[img_inp, state_a, state_b], outputs=[out_, a, b])


def nic(num_word=2000, dim_word=1056, lstm_units=2000, img_size=224, mode_train=True):
    if mode_train:
        weights = "imagenet"
    else:
        weights = None
    nasn_model = tf.keras.applications.nasnet.NASNetMobile(weights=weights)
    # nasn_model = Xception()
    img_input = nasn_model.input
    nasn_model = tf.keras.Model(inputs=[img_input],
                                outputs=[nasn_model.get_layer("global_average_pooling2d").output])
    nasn_model.trainable = False
    img_inp = tf.keras.Input(shape=(img_size, img_size, 3))
    # print(nasn_model.output)
    # print(img_embedding)
    img_embedding = tf.keras.layers.Dense(dim_word, activation="relu")(nasn_model(img_inp))
    img_embedding = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, -2))(img_embedding)
    word_input = tf.keras.Input(shape=[None])
    word_embedding_layer = tf.keras.layers.Embedding(num_word, dim_word)
    # word_embedding_layer.trainable = False
    word_embedding = word_embedding_layer(word_input)
    # print(word_embedding)
    decode_input = tf.keras.layers.Lambda(lambda x:tf.concat(x, axis=-2))([img_embedding, word_embedding])
    # print(decode_input)
    decode = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
    decode_output, _, __ = decode(decode_input)
    # decode_2 = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
    # decode_output_2 = decode_2(decode_output)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_word, activation="softmax"))(decode_output)
    model = tf.keras.Model(inputs=[img_inp, word_input], outputs=[outputs])
    # model.summary()
    return model
