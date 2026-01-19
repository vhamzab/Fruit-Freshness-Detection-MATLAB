# 🍎 Meyve ve Sebze Tazelik Tespit Sistemi (Fruit Freshness Detection System)

![MATLAB](https://img.shields.io/badge/Platform-MATLAB-orange)
![Deep Learning](https://img.shields.io/badge/Method-Deep%20Learning%20%26%20SVM-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-99.60%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> **English Summary:** This project is a real-time fruit freshness classification system developed using MATLAB. It compares Traditional Machine Learning (Bag of Features) and Deep Learning (AlexNet) methods. The final model achieves **99.60% accuracy** across 18 classes and includes a user-friendly GUI with a stabilization algorithm for live webcam testing.

---

## 📖 Proje Hakkında
Bu projenin temel amacı, tarım ve gıda endüstrisinde insan hatasını en aza indirmek için **Yapay Zeka** destekli otomatik bir kalite kontrol sistemi geliştirmektir. 

Sistem; Elma, Muz, Portakal, Salatalık gibi 9 farklı ürünün taze mi yoksa çürük mü olduğunu statik görsellerden veya **canlı kamera** görüntüsünden tespit edebilmektedir.

### 🎯 Özellikler
* **9 Farklı Meyve/Sebze Türü:** 18 Sınıflı geniş kapsamlı tespit (Fresh/Rotten).
* **İki Farklı Yöntem:** Geleneksel (SVM) ve Modern (Deep Learning) yaklaşımların karşılaştırılması.
* **Gerçek Zamanlı Tespit:** Webcam üzerinden anlık görüntü işleme.
* **Akıllı ROI (İlgi Alanı):** Arka plan gürültüsünü engelleyen odaklanma algoritması.
* **Stabilizasyon:** Hareketli ortalama filtresi ile titremesiz, kararlı sonuçlar.
* **GUI:** MATLAB App Designer ile geliştirilmiş endüstriyel arayüz.

---

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

Bu çalışmada iki ana yaklaşım kıyaslanmıştır:

### 1️⃣ Yöntem A: Geleneksel Makine Öğrenmesi (Machine Learning)
* **Özellik Çıkarımı:** SURF (Speeded-Up Robust Features)
* **Sözlük Oluşturma:** K-Means Kümeleme (Bag of Features)
* **Sınıflandırma:** Çok Sınıflı SVM (Support Vector Machine)
* *Sonuç:* Özellikle şekli benzer meyvelerde (Elma/Portakal) karışıklık yaşanmış, başarı **%69.35** seviyesinde kalmıştır.

### 2️⃣ Yöntem B: Derin Öğrenme (Deep Learning) - **(Kazanan Yöntem)**
* **Model:** AlexNet (Transfer Learning)
* **Optimizasyon:** SGDM (Stochastic Gradient Descent with Momentum)
* **Eğitim:** 1.2 Milyon görüntü ile ön eğitimli model, proje veri setine (20.000+ görüntü) uyarlanmıştır (Fine-tuning).
* *Sonuç:* Doku ve renk detaylarını öğrenerek **%99.60** başarıya ulaşılmıştır.

---

## 📊 Performans Karşılaştırması

| Yöntem | Algoritma | Başarı Oranı (Accuracy) | Yorum |
| :--- | :--- | :--- | :--- |
| **Yöntem A** | Bag of Features (SVM) | %69.35 | Şekil odaklı, düşük performans. |
| **Yöntem B** | **Deep Learning (AlexNet)** | **%99.60** 🏆 | Doku ve renk odaklı, yüksek performans. |

---

## 🖥️ Arayüz ve Algoritmalar

Proje, son kullanıcı için **MATLAB App Designer** kullanılarak görselleştirilmiştir.

### 🔍 ROI (Region of Interest) Algoritması
Kameranın tüm odayı taraması yerine, sadece merkezdeki **300x300** piksellik alana odaklanması sağlanmıştır. Bu sayede uzaktaki nesneler bile yüksek doğrulukla tespit edilir.

### ⚖️ Stabilizasyon (Smoothing) Algoritması
Canlı yayındaki ışık değişimlerinden kaynaklanan "titremeyi" önlemek için aşağıdaki formül kullanılmıştır:

$$Guven_{yeni} = (Guven_{eski} \times 0.7) + (AnlikSkor \times 0.3)$$

Bu sayede ibre ve sonuç yazısı anlık değişimlerden etkilenmez, kararlı bir ölçüm sunar.

---


## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için:

1.  Bu repoyu indirin:
    ```bash
    git clone [https://github.com/kullaniciadi/Meyve-Tazelik-Tespiti.git](https://github.com/kullaniciadi/Meyve-Tazelik-Tespiti.git)
    ```
2.  **MATLAB**'ı açın (R2020a veya üzeri önerilir).
3.  Gerekli eklentilerin yüklü olduğundan emin olun:
    * *Deep Learning Toolbox*
    * *Computer Vision Toolbox*
    * *AlexNet Support Package*
4.  `MeyveGUI.mlapp` dosyasını çalıştırın.
5.  Eğer eğitilmiş model (`FinalProjeModelim.mat`) klasördeyse sistem direkt açılacaktır.

---

## 📚 Referanslar

1.  *Krizhevsky, A., et al. (2012).* Imagenet classification with deep convolutional neural networks.
2.  *Muresan, H., & Oltean, M. (2018).* Fruit recognition from images using deep learning.
3.  *Mohanty, S. P., et al. (2016).* Using deep learning for image-based plant disease detection.
4. *Naranjo-Torres, J., Mora, M., Hernández-García, R., Barrientos, R. J., Fredes, C., & Valenzuela, A. (2020). A review of convolutional neural network applied to fruit image processing. Applied Sciences, 10(10), 3443.
5. *Dubey, S. R., & Jalal, A. S. (2016). Apple disease classification using color, texture and shape features from images. Signal, Image and Video Processing, 10(5), 819-826.
---

**Geliştiriciler:** [vAHİT HAMZA BARAN] & [NURAN ERGENÇ]