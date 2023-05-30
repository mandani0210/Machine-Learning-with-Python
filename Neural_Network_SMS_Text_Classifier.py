import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.data_utils import pad_sequences

TRAIN_DATA_URL = "https://raw.githubusercontent.com/beaucarnes/fcc_python_curriculum/master/sms/train-data.tsv"
TEST_DATA_URL  = "https://raw.githubusercontent.com/beaucarnes/fcc_python_curriculum/master/sms/valid-data.tsv"

train_file_path = tf.keras.utils.get_file("train-data.tsv", TRAIN_DATA_URL)
test_file_path  = tf.keras.utils.get_file("valid-data.tsv", TEST_DATA_URL)

df_train = pd.read_csv(train_file_path, sep="\t", header=None, names=['y', 'x'])
df_train.head()

df_test = pd.read_csv(test_file_path, sep="\t", header=None, names=['y', 'x'])
df_test.head()

y_train = df_train['y'].astype('category').cat.codes
y_test  = df_test['y'].astype('category').cat.codes
y_train[:5]

bar = df_train['y'].value_counts()
plt.bar(bar.index, bar)
plt.xlabel('Label')
plt.title('Number of ham and spam messages')

import nltk
nltk.download('stopwords') # download stopwords
nltk.download('wordnet')   # download vocab for lemmatizer

import re
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords'

lemmatizer = WordNetLemmatizer()
def clean_txt(txt):
    txt = re.sub(r'([^\s\w])+', ' ', txt)
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()
                    if not word in stopwords_eng])
    txt = txt.lower()
    return txt
  
X_train = df_train['x'].apply(lambda x: clean_txt(x))
X_train[:5]
 
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing import sequence
  
# Keep top 1000 frequently occurring words
max_words = 1000
# Cut off the words after seeing 500 words in each document
max_len = 500

t = Tokenizer(num_words=max_words)
t.fit_on_texts(X_train)
  
sequences = t.texts_to_sequences(X_train)
sequences[:5]  
  
sequences_matrix = tf.keras.utils.pad_sequences(sequences, maxlen=max_len)
sequences_matrix[:5]
  
i = tf.keras.layers.Input(shape=[max_len])
x = tf.keras.layers.Embedding(max_words, 50, input_length=max_len)(i)
x = tf.keras.layers.LSTM(64)(x)

x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='relu')(x)

model = tf.keras.models.Model(inputs=i, outputs=x)
model.compile(
    loss='binary_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)
model.summary()
  
r = model.fit(sequences_matrix, y_train,
              batch_size=128, epochs=10,
              validation_split=0.2, #)
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', min_delta=0.0001)])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

def preprocessing(X):
  x = X.apply(lambda x: clean_txt(x))
  x = t.texts_to_sequences(x)
  return sequence.pad_sequences(x, maxlen=max_len)

s = model.evaluate(preprocessing(df_test['x']), y_test)

def predict_message(pred_text):
  p = model.predict(preprocessing(pd.Series([pred_text])))[0]

  return (p[0], ("kamno msg chhe 😄" if p<0.5 else "laa aato spam chhe 🤦‍♀️"))

# pred_text = "Use pdfFiller for your paperless document management Easily edit and annotate PDFs as well as eSign documents and send them for signing. Perform even more actions using pdfFiller smart features: Collect client data using fillable forms. Make your document accessible online. Get payments by connecting the Stripe payment gateway to the documents you share. Collect legally binding eSignatures from your customers and partners. Optimize your team’s collaboration. Invite up to 4 users to share a single pdfFiller account. Start your free 30-day trial and join the company of thousands of professionals: Start free trial Cancel your free trial any time for any reason!"
pred_text = input("enter string or msg::")
prediction = predict_message(pred_text)
print(prediction)










  
  

