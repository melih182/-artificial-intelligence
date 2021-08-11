import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# "train" isimli klasörde yer alan resimleri eğitimde kullanacağız
EGITIM_KLASORU = 'train'

# "test" isimli klasörde yer alan resimleri testte kullanacağız
TEST_KLASORU = 'test1'

# eğitim ve test için kullanılan resimleri aynı boyuta (50x50 piksel) getireceğiz.
RESIM_BOYUTU = 50

# öğrenme oranını 0.001 olarak tanımlayalım.
OGRENME_ORANI = 1e-3

# oluşturacağımız modele bir isim verelim.
MODEL_ADI = 'kedi-kopek-ayirici'

### DOSYA ADLARINDAN ETİKET BİLGİLERİNİN ALINMASI ###

# etiket_olustur isminde bir fonksiyon tanımlayalım.
# bu fonksiyon ile dosya adlarında yer alan "cat" ya da "dog" etiketlerini algılayacağız.
# fonksiyon dosya adı "cat" ise [1 0], "dog" ise [0 1] çıkışını verir.

def etiket_olustur(resim_adi):
    obje_turu = resim_adi.split('.')[-3]  # dosya adında bulunan "cat" ya da "dog" kelimesini al 
    if obje_turu == 'cat':
        return np.array([1, 0])
    elif obje_turu == 'dog':
        return np.array([0, 1])

### RESİMLERİN MATRİS HALİNE DÖNÜŞTÜRÜLMESİ ###

# train klasöründeki resimlerden eğitimde kullanılabilecek şekilde eğitim verisi oluştur.
# oluşturulan eğitim verisi "egitim_verisi.npy" isimli dosyaya yazılır.
# fonksiyon içerisinde verilerin karıştırılması (shuffle) sağlanır.
# resimler gri olarak okunup 50x50 piksel olacak şekilde yeniden boyutlandırılır.

def egitim_verisi_olustur():
    olusturulan_egitim_verisi = []
    for img in tqdm(os.listdir(EGITIM_KLASORU)):
        dosya_yolu = os.path.join(EGITIM_KLASORU, img)
        resim_verisi = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        resim_verisi = cv2.resize(resim_verisi, (RESIM_BOYUTU, RESIM_BOYUTU))
        olusturulan_egitim_verisi.append([np.array(resim_verisi), etiket_olustur(img)])
    shuffle(olusturulan_egitim_verisi)
    np.save('egitim_verisi.npy', olusturulan_egitim_verisi)
    return olusturulan_egitim_verisi

# test klasöründeki resimlerden eğitimde kullanılabilecek şekilde test verisi oluştur.
# oluşturulan test verisi "test_verisi.npy" isimli dosyaya yazılır
# fonksiyon içerisinde verilerin karıştırılması (shuffle) sağlanır.
# resimler gri olarak okunup 50x50 piksel olacak şekilde yeniden boyutlandırılır.

def test_verisi_olustur():
    olusturulan_test_verisi = []
    for img in tqdm(os.listdir(TEST_KLASORU)):
        dosya_yolu = os.path.join(TEST_KLASORU, img)
        resim_no = img.split('.')[0]
        resim_verisi = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        resim_verisi = cv2.resize(resim_verisi, (RESIM_BOYUTU, RESIM_BOYUTU))
        olusturulan_test_verisi.append([np.array(resim_verisi), resim_no])
    shuffle(olusturulan_test_verisi)
    np.save('test_verisi.npy', olusturulan_test_verisi)
    return olusturulan_test_verisi

# "egitim_verisi.npy" ve "test_verisi.npy" dosyaları daha önce oluşturulmadıysa:
# egitim_verisi = egitim_verisi_olustur()
# test_verisi = test_verisi_olustur()

# "egitim_verisi.npy" ve "test_verisi.npy" dosyaları oluşturulduysa:
egitim_verisi = np.load('egitim_verisi.npy',allow_pickle=True)
test_verisi = np.load('test_verisi.npy',allow_pickle=True)

# ağımızı eğitirken 5000 adet resmi eğitimi test etmek için kullanacağız.
egitim = egitim_verisi[:-5000]
test = egitim_verisi[-5000:]

X_egitim = np.array([i[0] for i in egitim]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1)
y_egitim = [i[1] for i in egitim]
X_test = np.array([i[0] for i in test]).reshape(-1, RESIM_BOYUTU, RESIM_BOYUTU, 1)
y_test = [i[1] for i in test]

# Veriseti görselleştirmesi
_, counts = np.unique(y_egitim, return_counts=True)
keys = ["dog","cat"]
plt.bar(keys, counts)
plt.show()

### MİMARİNİN OLUŞTURULMASI ###

tf.compat.v1.reset_default_graph()

# ağımızın girişinin boyutlarının ne olacağını tanımlayalım
convnet = input_data(shape=[None, RESIM_BOYUTU, RESIM_BOYUTU, 1], name='input')

# 32 adet 5x5 boyutunda filtrelerden oluşan ve relu aktivasyonlu konvolüsyon katmanı
convnet = conv_2d(convnet, 32, 5, activation='relu')

# 5x5 boyutunda filtelerden oluşan max_pool katmanı
convnet = max_pool_2d(convnet, 5)

# 64 adet 5x5 boyutunda filtrelerden oluşan ve relu aktivasyonlu konvolüsyon katmanı
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# 128 adet 5x5 boyutunda filtrelerden oluşan ve relu aktivasyonlu konvolüsyon katmanı
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# 64 adet 5x5 boyutunda filtrelerden oluşan ve relu aktivasyonlu konvolüsyon katmanı
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# 32 adet 5x5 boyutunda filtrelerden oluşan ve relu aktivasyonlu konvolüsyon katmanı
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# 1024 birimden oluşan tam bağlantılı ve relu aktivasyonlu katman
convnet = fully_connected(convnet, 1024, activation='relu')

# aşırı öğrenmeyi yani ezberlemeyi (overfitting) engellemek için dropout katmanı
convnet = dropout(convnet, 0.8)

# 2 birimli ve softmax aktivasyonlu tam bağlantılı katman
convnet = fully_connected(convnet, 2, activation='softmax')

# oluşturulan mimariyi, öğrenme oranını, optimizasyon türünü, kayıp fonksiyonunu ve dosya isimlerinden aldığımız hedef değerlerini
# kullanarak ağı oluşturalım.
convnet = regression(convnet, optimizer='adam', learning_rate=OGRENME_ORANI, loss='categorical_crossentropy',
                     name='targets')

# OLUŞTURULAN MİMARİ İLE DEEP LEARNING NETWORK (DNN) MODELİ OLUŞTURULMASI
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

# VERİLERLE EĞİTİM YAPILMASI
model.fit({'input': X_egitim}, {'targets': y_egitim}, n_epoch=10,
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=MODEL_ADI)

### OLUŞTURULAN DERİN AĞ MODELİNİN TEST VERİLERİ ÜZERİNDE DENENMESİ

fig = plt.figure(figsize=(16, 12))

for no, veri in enumerate(test_verisi[:16]):

    resim_no = veri[1]
    resim_verisi = veri[0]

    y = fig.add_subplot(4, 4, no + 1)
    orig = resim_verisi
    veri = resim_verisi.reshape(RESIM_BOYUTU, RESIM_BOYUTU, 1)
    ag_cikisi = model.predict([veri])[0]

    if np.argmax(ag_cikisi) == 1:
        str_label = 'Köpek'
    else:
        str_label = 'Kedi'

    y.imshow(orig, cmap='gray') 
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml') 
face_cascade2 = cv2.CascadeClassifier('dog_face.xml') 


cap = cv2.VideoCapture(0) 
### KAMERA AÇIKSA DÖNGÜ BAŞLANGICI
while 1: 
  
    ### KAMERADAN GÖRÜNTÜ OKUMA
    ret, img = cap.read() 
  
    ### HER BİR KAREYİ GRİ ÖLÇEKLENDİRME 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    ### EKRANDAKİ FARKLI BOYUTLARDAKİ YÜZLERİ ALGILAMA 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_cascade2.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        ### YÜZLERİ KARE İÇİNDE GÖSTERME 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2) 
        cv2.putText(img, 'Kedi', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
    for (x,y,w,h) in faces2: 
        ### YÜZLERİ KARE İÇİNDE GÖSTERME
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        cv2.putText(img, 'Kopek', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) 
        roi_gray2 = gray[y:y+h, x:x+w] 
        roi_color2 = img[y:y+h, x:x+w] 
  
    ### GÖRÜNTÜYÜ PENCEREDE YAYINLAMA 
    cv2.imshow('Kamera',img) 
  
    ### PROGRAMI DURDURMAK İÇİN ESC TUŞUNA BASILMASINI BEKLEME
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
### ESCYE BASILDIYSA PENCEREYİ KAPATMA
cap.release() 
  
### RAM KULLANIMINI DENGELEME 
cv2.destroyAllWindows() 1
