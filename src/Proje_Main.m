%% BÖLÜM 1: Kurulum ve Veri Yükleme
% ekranı temizle
clc; clear; close all;

% Veri Seti Yolunu Tanımla
% not:'VeriSeti' klasörü bu kodun olduğu klasörde olmalı
veriKlasoru = 'VeriSeti'; 

% imageDatastore Oluşturma
% imageDatastore, binlerce resmi belleği şişirmeden yönetmemizi sağlar.
% 'IncludeSubfolders' -> Alt klasörleri de oku
% 'LabelSource' -> Klasör isimlerini etiket (Sınıf adı) olarak kullan
imds = imageDatastore(veriKlasoru, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Veri Seti Özeti
disp('--- Veri Seti Bilgileri ---');
kategoriSayilari = countEachLabel(imds);
disp(kategoriSayilari);

% 4. Eğitim ve Test Olarak Ayırma (%70 Eğitim, %30 Test)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

disp(['Eğitim görsel sayısı: ', num2str(numel(imdsTrain.Files))]);
disp(['Test görsel sayısı: ', num2str(numel(imdsTest.Files))]);

% örnek görselleri gösterme
numImages = 8;
perm = randperm(numel(imdsTrain.Files), numImages);
figure('Name', 'Veri Setinden Ornekler');
for i = 1:numImages
    subplot(2, 4, i);
    imshow(imdsTrain.Files{perm(i)});
    [~, name] = fileparts(imdsTrain.Files{perm(i)});
    % Klasör adını (etiketi) başlığa yaz
    title(char(imdsTrain.Labels(perm(i))));
end


%% BÖLÜM 2: Yöntem A - Bag of Features (Geleneksel Yöntem)
disp('--- Bag of Features İşlemi Başlıyor (Biraz zaman alabilir) ---');

% Görsel Kelime Çantasını (Bag) Oluşturma
% 'VocabularySize', 500 -> 500 farklı özellik grubu oluşturur.
bag = bagOfFeatures(imdsTrain, ...
    'VocabularySize', 500, ...
    'PointSelection', 'Detector');

disp('Görsel özellikler çıkarıldı. Model eğitiliyor...');

% Çoklu Sınıf SVM Eğitimi (Classifier Training)
% Çıkarılan özellikleri kullanarak kategorileri öğrenir.
categoryClassifier = trainImageCategoryClassifier(imdsTrain, bag);

disp('Model eğitimi tamamlandı. Test ediliyor...');

% Modeli Test Setiyle Değerlendirme
% Bu fonksiyon otomatik olarak tahmin yapar ve Konfüzyon Matrisi oluşturur.
confMatrix = evaluate(categoryClassifier, imdsTest);

% Ortalama Doğruluk (Accuracy) Hesaplama
meanAccuracy = mean(diag(confMatrix));
disp(['Yöntem A Başarısı (Ortalama Doğruluk): %', num2str(meanAccuracy * 100)]);


%% BÖLÜM 3: Yöntem B - Derin Öğrenme (Transfer Learning)
disp('--- Derin Öğrenme (Transfer Learning) Başlıyor ---');

% Hazır Ağı Yükle (AlexNet kullanılacak)
try
    net = alexnet; 
catch
    error('AlexNet yüklü değil.');
end

% Ağın Giriş Katmanını Ayarla
% AlexNet 227x227 boyutunda resim ister. Verileri buna göre yeniden boyutlandıralım.
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Sınıf sayısını (18) klasörlerden otomatik öğrensin:
numClasses = numel(categories(imdsTrain.Labels)); 

layers = [
    imageInputLayer([227 227 3])
    net.Layers(2:end-3)
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Eğitim Seçeneklerini Belirle

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 3, ... % Eğitim tur sayısı
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress'); % Ekranda canlı grafik açar

% Yeni Modeli Eğit
disp('Ağ eğitiliyor, lütfen bekleyin (Grafik penceresi açılacak)...');
[trainedNet, trainInfo] = trainNetwork(augimdsTrain, layers, options);

% Test ve Performans
predictedLabels = classify(trainedNet, augimdsTest);
accuracyDeep = mean(predictedLabels == imdsTest.Labels);

disp(['Yöntem B Başarısı (Deep Learning): %', num2str(accuracyDeep * 100)]);

% Karşılaştırma Grafiği Çiz
figure('Name', 'Yontem Karsilastirmasi');
bar([meanAccuracy * 100, accuracyDeep * 100]);
set(gca, 'XTickLabel', {'Bag of Features', 'Deep Learning'});
ylabel('Doğruluk (%)');
title('Model Performans Karşılaştırması');
text(1, meanAccuracy*100, [num2str(round(meanAccuracy*100)), '%'], 'Vert','bottom','Horiz','center');
text(2, accuracyDeep*100, [num2str(round(accuracyDeep*100)), '%'], 'Vert','bottom','Horiz','center');


%% BÖLÜM 4: RESİM YÜKLEME TESTİ (POP-UP MESAJ KUTULU)
clc; close all;
% Model kontrolü
if ~exist('trainedNet', 'var'), load('FinalProjeModelim.mat'); end

% Dosya Seç
[dosya, yol] = uigetfile({'*.jpg;*.png;*.jpeg', 'Resim Dosyalari'});
if isequal(dosya,0), return; end

% Resmi Oku ve Göster
img = imread(fullfile(yol, dosya));
imgResized = imresize(img, [227 227]);

% Resmi hemen göster (İşlemden önce)
figure('Name', 'Secilen Resim', 'NumberTitle', 'off');
imshow(img);
title('Analiz Ediliyor...', 'FontSize', 14);
drawnow;

% Tahmin Et
[label, scores] = classify(trainedNet, imgResized);
guven = max(scores) * 100;
etiketStr = char(label);

% TAM TÜRKÇE SÖZLÜK (Hatasız Çeviri İçin)
if contains(etiketStr, 'freshapples'), urun='TAZE ELMA'; Durum='TAZE';
elseif contains(etiketStr, 'rottenapples'), urun='CURUK ELMA'; Durum='CURUK';
elseif contains(etiketStr, 'freshbanana'), urun='TAZE MUZ'; Durum='TAZE';
elseif contains(etiketStr, 'rottenbanana'), urun='CURUK MUZ'; Durum='CURUK';
elseif contains(etiketStr, 'freshoranges'), urun='TAZE PORTAKAL'; Durum='TAZE';
elseif contains(etiketStr, 'rottenoranges'), urun='CURUK PORTAKAL'; Durum='CURUK';
elseif contains(etiketStr, 'freshtomato'), urun='TAZE DOMATES'; Durum='TAZE';
elseif contains(etiketStr, 'rottentomato'), urun='CURUK DOMATES'; Durum='CURUK';
elseif contains(etiketStr, 'freshpotato'), urun='TAZE PATATES'; Durum='TAZE';
elseif contains(etiketStr, 'rottenpotato'), urun='CURUK PATATES'; Durum='CURUK';
elseif contains(etiketStr, 'freshcucumber'), urun='TAZE SALATALIK'; Durum='TAZE';
elseif contains(etiketStr, 'rottencucumber'), urun='CURUK SALATALIK'; Durum='CURUK';
elseif contains(etiketStr, 'freshcapsicum'), urun='TAZE BIBER'; Durum='TAZE';
elseif contains(etiketStr, 'rottencapsicum'), urun='CURUK BIBER'; Durum='CURUK';
elseif contains(etiketStr, 'freshokra'), urun='TAZE BAMYA'; Durum='TAZE';
elseif contains(etiketStr, 'rottenokra'), urun='CURUK BAMYA'; Durum='CURUK';
elseif contains(etiketStr, 'freshbittergroud'), urun='TAZE KUDRET NARI'; Durum='TAZE';
elseif contains(etiketStr, 'rottenbittergroud'), urun='CURUK KUDRET NARI'; Durum='CURUK';
else, urun = 'TANIMLANAMADI'; Durum='BELIRSIZ';
end

% SONUCU GÖSTER (Resim üzerine değil, ekrana kutu olarak aç)
mesaj = sprintf('TESPIT EDILEN: %s\nDURUM: %s\nGUVEN ORANI: %%%.1f', urun, Durum, guven);
title(urun, 'FontSize', 16, 'Color', 'blue', 'FontWeight', 'bold'); % Başlığı güncelle

% Sonucu Ekrana Patlat (En garanti yöntem)
msgbox(mesaj, 'Analiz Sonucu'); 
disp(['Sonuç: ' urun]);

%% BÖLÜM 5: KAMERA İLE CANLI TEST
clc; close all;
if ~exist('trainedNet', 'var'), load('FinalProjeModelim.mat'); end

cam = webcam;
hFig = figure('Name', 'MEYVE AYIKLAMA SİSTEMİ TEST PENCERESİ', 'NumberTitle', 'off', 'MenuBar', 'none');

while ishandle(hFig)
    try
        img = snapshot(cam);
        [h, w, ~] = size(img);
        cropRect = [(w-300)/2, (h-300)/2, 300, 300]; 
        imgCrop = imcrop(img, cropRect);
        imgResized = imresize(imgCrop, [227 227]);
        
        [label, scores] = classify(trainedNet, imgResized);
        confidence = max(scores) * 100;
        etiketStr = char(label);
        
        % RENK VE İSİM AYARLAMA (MANUEL SÖZLÜK)
        urunIsmi = 'Tanimlaniyor...';
        boxColor = 'yellow';
        
        if confidence > 50
            if contains(etiketStr, 'freshapples'), urunIsmi='TAZE ELMA'; boxColor='green';
            elseif contains(etiketStr, 'rottenapples'), urunIsmi='CURUK ELMA'; boxColor='red';
            elseif contains(etiketStr, 'freshbanana'), urunIsmi='TAZE MUZ'; boxColor='green';
            elseif contains(etiketStr, 'rottenbanana'), urunIsmi='CURUK MUZ'; boxColor='red';
            elseif contains(etiketStr, 'freshoranges'), urunIsmi='TAZE PORTAKAL'; boxColor='green';
            elseif contains(etiketStr, 'rottenoranges'), urunIsmi='CURUK PORTAKAL'; boxColor='red';
            elseif contains(etiketStr, 'freshtomato'), urunIsmi='TAZE DOMATES'; boxColor='green';
            elseif contains(etiketStr, 'rottentomato'), urunIsmi='CURUK DOMATES'; boxColor='red';
            elseif contains(etiketStr, 'freshpotato'), urunIsmi='TAZE PATATES'; boxColor='green';
            elseif contains(etiketStr, 'rottenpotato'), urunIsmi='CURUK PATATES'; boxColor='red';
            elseif contains(etiketStr, 'freshcucumber'), urunIsmi='TAZE SALATALIK'; boxColor='green';
            elseif contains(etiketStr, 'rottencucumber'), urunIsmi='CURUK SALATALIK'; boxColor='red';
            elseif contains(etiketStr, 'freshcapsicum'), urunIsmi='TAZE BIBER'; boxColor='green';
            elseif contains(etiketStr, 'rottencapsicum'), urunIsmi='CURUK BIBER'; boxColor='red';
            elseif contains(etiketStr, 'freshokra'), urunIsmi='TAZE BAMYA'; boxColor='green';
            elseif contains(etiketStr, 'rottenokra'), urunIsmi='CURUK BAMYA'; boxColor='red';
            elseif contains(etiketStr, 'freshbittergroud'), urunIsmi='TAZE KUDRET NARI'; boxColor='green';
            elseif contains(etiketStr, 'rottenbittergroud'), urunIsmi='CURUK KUDRET NARI'; boxColor='red';
            end
        else
            urunIsmi = 'Nesne Bekleniyor...';
            boxColor = 'yellow';
        end
        
        % ÇİZİM
        img = insertShape(img, 'Rectangle', cropRect, 'LineWidth', 5, 'Color', boxColor);
        yazi = sprintf('%s (%%%0.1f)', urunIsmi, confidence);
        
        % Yazı sol üste
        img = insertText(img, [10 10], yazi, 'FontSize', 24, 'BoxColor', boxColor, 'TextColor', 'black');
            
        imshow(img);
        drawnow;
    catch
        break;
    end
end
clear cam; close all;