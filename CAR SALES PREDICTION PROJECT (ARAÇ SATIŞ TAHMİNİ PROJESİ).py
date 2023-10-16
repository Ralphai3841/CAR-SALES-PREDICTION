#!/usr/bin/env python
# coding: utf-8

# #  CAR SALES PREDICTION PROJECT (ARAÇ SATIŞ TAHMİNİ PROJESİ)

# ## INTRODUCTİON (GİRİŞ)
# 
# You are working as a car salesman and you would like to develop a model to predict the total amount that customers are willing to pay given the following attributes: 
# (Bir araba satıcısı olarak çalışıyorsunuz ve aşağıdaki özellikler göz önüne alındığında müşterilerin ödemeye hazır oldukları toplam tutarı tahmin etmek için bir model geliştirmek istiyorsunuz:)
# - Customer Name (Müşteri Adı)
# - Customer e-mail (Müşteri E-maili)
# - Country (Ülke-Bölge)
# - Gender (Cinsiyet)
# - Age  (Yaş)
# - Annual Salary  (Yıllık Gelir)
# - Credit Card Debt (Kredi Kartı Borcu)
# - Net Worth (Net Değer)
# 
# The model should predict: (Modelin Tahmin Edeceği Değer)
# - Car Purchase Amount  (Araç Alım Tutarı)

#  ## Libraries Import        (Kütüphanelerin import edilmesi)    
#  
# ### We import the libraries we will use in the project. We import the libraries we will use in the project to analyze and visualize the data set. 
# ### (Projede kullanacağımız kütüphaneleri import ederiz.Projede kullanacağımız kütüphaneleri veri setini analiz etmek ve görselleştirmek için import ederiz.)

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Import Dataset          (Verisetinin import edilmesi)

# In[3]:


car_df = pd.read_csv('C:/Users/ralpd/OneDrive/Masaüstü/Yeni klasör/Yeni klasör/CAR_SALES_PREDICTION_PROJECT_(ARAÇ_SATIŞ_TAHMİNİ_PROJESİ).csv',
             encoding="ISO-8859-1")


# In[4]:


car_df


# ## Visualize Dataset (Veri setinin görselleştirilmesi)

# In[5]:


sns.pairplot(car_df)


# ## Create Testing and Training Dataset/Data Cleaning  (Test ve Eğitim Veri Seti / Veri Temizliği)
# 
# ###  We prepare the Data Set. Our purpose in performing Data Cleaning is to ensure that the data is prepared before starting data analysis and that the data to be analyzed is brought to its final form.
# ### (Veri Setini hazırlarız. Veri Temizliği yapmaktaki amacımız veri analizine başlamadan önce verinin hazırlanmasını ve analiz edilecek datanın son şekline getirilmesini sağlamaktır.)

# In[6]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[ ]:





# In[7]:


X


# In[8]:


y = car_df['Car Purchase Amount']
y.shape


# In[9]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[10]:


scaler.data_max_


# In[11]:


scaler.data_min_


# In[25]:


print(X_scaled[:,0])


# In[12]:


y.shape


# In[13]:


y = y.values.reshape(-1,1)


# In[14]:


y.shape


# In[15]:


y_scaled = scaler.fit_transform(y)


# In[30]:


y_scaled


# ## Training The Model (Modelin Eğitilmesi)
# 
# ### Bu kısımda modelin train edilmesi için ML ve DL kütüphanelerinden Scikit Learn, Tensorflow ve Keras kütüphanelerini kullanırız.
# 
# ### (In this part, we use Scikit Learn, Tensorflow and Keras libraries from ML and DL libraries to train the model.)

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[17]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[18]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[19]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)


# ## Evaluating The Model (Modelin Değerlendirilmesi)

# In[20]:


print(epochs_hist.history.keys())


# In[21]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[22]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])


# In[23]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[24]:


print('Expected Purchase Amount=', y_predict[:,0])


# In[ ]:




