# 🧠 Beyin Tümörü Tespit Uygulaması

Bu proje, beyin tümörlerinin tespit edilmesi için bir derin öğrenme modeli kullanmaktadır. Model, MRI görüntülerinden tümör olup olmadığını tespit ederek sınıflandırma yapmaktadır. Derin öğrenme teknikleri kullanarak sağlık alanında hızlı ve doğru teşhis imkanı sağlamayı amaçlamaktadır.

## 📌 Proje İçeriği

- **`Brain_Tumor_Detection_Train.py`**: Verisetini kullanarak CNN modelini eğiten ve kaydeden dosya.
- **`Brain_Tumor_Detection_Test.py`**: Eğitilmiş modeli kullanarak yeni görüntüler üzerinde tahmin yapan dosya.
- **`BrainTumor150EpochsCategorical.keras`**: En iyi sonucu veren derin öğrenme modelini içeren dosya.
- **MRI Görüntü Verisi**: Beyin tümörü teşhisi için kullanılan gerçek MRI görüntülerinden oluşan veri seti.

## 🚀 Kurulum ve Kullanım

### 1️⃣ Gereksinimleri Yükleyin
Aşağıdaki komutları çalıştırarak gerekli kütüphaneleri yükleyin:

```bash
pip install numpy pandas keras tensorflow opencv-python matplotlib seaborn natsort pillow scikit-learn
```

### 2️⃣ Modeli Eğitme
`Brain_Tumor_Detection_Train.py` dosyasını çalıştırarak modeli eğitin:

```bash
python Brain_Tumor_Detection_Train.py
```

Model eğitildikten sonra `BrainTumor150EpochsCategorical.keras` olarak kaydedilir. Model, 150 epoch boyunca eğitimden geçirilerek en iyi sonuçları verecek şekilde optimize edilmiştir.

### 3️⃣ Tahmin Yapma
Eğitilmiş modeli kullanarak yeni MRI görüntülerini sınıflandırmak için:

```bash
python Brain_Tumor_Detection_Test.py
```

Model, MRI görüntülerini analiz eder ve tümörlü olup olmadığını tahmin eder. Sonuçlar, doğruluk metrikleri ve görselleştirmeler ile desteklenmektedir.

## 📊 Kullanılan Model ve Algoritmalar

Bu projede **Convolutional Neural Networks (CNN)** kullanılmıştır. Model aşağıdaki katmanlardan oluşmaktadır:

- **Conv2D Katmanları**: Görüntülerin kenar ve özelliklerini belirlemek için.
- **MaxPooling2D**: Boyut azaltma ve önemli özellikleri koruma.
- **Flatten ve Dense Katmanları**: Görüntüden elde edilen özellikleri sınıflandırmak için.
- **Activation: ReLU ve Softmax**: ReLU aktivasyon fonksiyonu ile derin öğrenme katmanlarında doğrusal olmayan dönüşümler yapılırken, son katmanda Softmax fonksiyonu ile tahminler elde edilmektedir.
- **Dropout**: Aşırı öğrenmeyi önlemek için rastgele bağlantıları kapatan mekanizma kullanılmıştır.

## 🔍 Veri Kümesi

Bu proje MRI görüntülerinden oluşan bir veri seti kullanmaktadır. Veri seti aşağıdaki iki sınıfı içermektedir:

- **Yes**: Beyin tümörü bulunan görüntüler.
- **No**: Beyin tümörü bulunmayan görüntüler.

Görüntüler **64x64 piksel** boyutuna dönüştürülerek modele beslenmektedir. Veri seti üzerinde ön işleme adımları uygulanarak modelin doğruluğu artırılmıştır.

## 📈 Model Performansı ve Sonuçlar

Modelin başarısı çeşitli metrikler ile değerlendirilmiştir:

- **Doğruluk (Accuracy)**: Modelin genel başarı oranı.
- **Hassasiyet (Sensitivity - Recall)**: Tümörlü vakaların doğru tespit edilme oranı.
- **Özgüllük (Specificity)**: Tümörsüz vakaların doğru tespit edilme oranı.
- **F1-Skoru**: Modelin denge performansını ölçen metrik.
- **Karışıklık Matrisi (Confusion Matrix)**: Modelin doğru ve yanlış tahminlerini gösteren matris.

Bu metrikler doğrultusunda modelin geliştirilmesi ve optimizasyonu sağlanmaktadır.

## 🌟 Olası Geliştirmeler

- Modelin **daha derin CNN yapıları ile geliştirilmesi**.
- **Veri setinin genişletilmesi** ile model performansının artırılması.
- Transfer öğrenme kullanılarak **VGG16, ResNet gibi** önceden eğitilmiş modellerin entegre edilmesi.
- **Gerçek zamanlı teşhis sistemleri** için geliştirme yapılması.
