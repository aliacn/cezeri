import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.patches import Rectangle

st.set_option('deprecation.showPyplotGlobalUse', False)

# Tarayıcı başlığını belirleme
favicon_path = "favicon.ico"
st.set_page_config(page_title="Breast Cancer Wisconsin (Diagnostic) Veri Seti Analizi",page_icon=favicon_path)

# Veri setini yükleme fonksiyonu
def load_data(file):
    data = pd.read_csv(file)
    return data

# Veriyi temizleme ve ön işleme adımları
def preprocess_data(data):
    # Gereksiz sütunları temizleme
    cleaned_data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # 'diagnosis' sütununu dönüştürme
    cleaned_data['diagnosis'] = cleaned_data['diagnosis'].map({'M': 1, 'B': 0})
    
    return cleaned_data

# Korelasyon matrisini çizme
def draw_correlation_matrix(data):
    malignant_data = data[data['diagnosis'] == 1]
    benign_data = data[data['diagnosis'] == 0]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, label='kotu', color='pink', edgecolor='red', palette="deep",s=100,alpha=0.5)
    sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, label='iyi', color='lightgreen', edgecolor='green',s=100,alpha=0.5)
    plt.xlabel('radius_mean')
    plt.ylabel('texture_mean')
    plt.title('Radius Mean - Texture Mean')
    plt.legend()
    st.pyplot()

# Veriyi ayırma
def split_data(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

 # KNN modelini oluşturma
def train_knn(X_train, y_train, n_neighbors=5, weights='uniform'):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn_model.fit(X_train, y_train)
    return knn_model

# KNN modeli için GridSearchCV ile en iyi parametreleri bulma
def find_best_knn_params(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

# En iyi parametrelerle KNN modelini eğitme
def train_knn_with_best_params(X_train, y_train, best_params):
    knn_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    knn_model.fit(X_train, y_train)
    return knn_model

# SVM modelini eğitme
def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_model.fit(X_train, y_train)
    return svm_model

# SVM modeli için GridSearchCV ile en iyi parametreleri bulma
def find_best_svm_params(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    param_grid = {'gamma': ['scale', 'auto']}
    svm_model = SVC(kernel=kernel, C=C)
    grid_search = GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_params['kernel'] = kernel
    best_params['C'] = C
    return best_params


# En iyi parametrelerle SVM modelini eğitme
def train_svm_with_best_params(X_train, y_train, best_params):
    svm_model = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'])
    svm_model.fit(X_train, y_train)
    return svm_model

# Naïve Bayes modelini eğitme
def train_naive_bayes(X_train, y_train):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model



#   Görev  4      /////////////////////////////////////
# Model sonuçlarını hesaplama
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, confusion

# Model sonuçlarını gösterme
def show_metrics(accuracy, precision, recall, f1, confusion):
    st.subheader("Model Sonuçları")
    st.write("Doğruluk (Accuracy):", accuracy)
    st.write("Kesinlik (Precision):", precision)
    st.write("Duyarlılık (Recall):", recall)
    st.write("F1-Skoru (F1-Score):", f1)
    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    
    # Karışıklık matrisi görselleştirme
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(confusion, cmap='magma', aspect='auto')
    
    # Color bar'ı 0-100 arasında 20'şer adımlarla oluştur
    cbar = plt.colorbar(img, ax=ax, ticks=np.arange(0, 101, 20))
    cbar.set_label('Percentage')

    # Heatmap oluşturma
    sns.heatmap(confusion, annot=True, fmt='d', cmap='magma', cbar=False, ax=ax)
    
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    ax.set_title('Karışıklık Matrisi')
    ax.set_xticks([0.5, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(['0', '1'])
    
    # Grafik çerçevesini kırmızı yapma
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.spines['bottom'].set_color('red')
    ax.spines['left'].set_color('red')
    
    # Arka plan rengini ayarlama
    ax.patch.set_facecolor('black')
    
    st.pyplot()


# Model sonuçlarını hesapla ve göster
def evaluate_model(y_true, y_pred):
    accuracy, precision, recall, f1, confusion = calculate_metrics(y_true, y_pred)
    show_metrics(accuracy, precision, recall, f1, confusion)

#/////////////////////////////////////////////////////

# Streamlit uygulaması
def main():
    st.title("Cezeri - Veri Temizleme ve Ön İşleme")
    uploaded_file = st.sidebar.file_uploader("Veri setini seçiniz", type=["csv"])
    
    # Sidebar'da model seçimi için bir radio düğmesi ekleme
    selected_model = st.sidebar.radio("Lütfen bir model seçin:", ["KNN", "SVM", "Naïve Bayes"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        cleaned_data = preprocess_data(data)
        
        st.subheader("Verinin Son 10 Satırı:")
        st.write(cleaned_data.tail(10))
        
        draw_correlation_matrix(cleaned_data)
        
        X_train, X_test, y_train, y_test = split_data(cleaned_data)
        st.write("X_train:", X_train.shape)
        st.write("X_test:", X_test.shape)
        st.write("Y_train:", y_train.shape)
        st.write("Y_test:", y_test.shape)
        
        if selected_model == "KNN":
            # KNN modelini eğitme
            st.subheader("KNN Modeli")
            best_params = find_best_knn_params(X_train, y_train)
            st.write("En iyi parametreler:", best_params)
            knn_model = train_knn_with_best_params(X_train, y_train, best_params)
            
            # Test verisi üzerinde modeli değerlendirme
            y_pred = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Test verisi doğruluk oranı:", accuracy)
            
            # Modeli değerlendirme
            evaluate_model(y_test, y_pred)

        elif selected_model == "SVM":
            # SVM modelini eğitme
            st.subheader("SVM Modeli")
            best_params = find_best_svm_params(X_train, y_train)
            st.write("En iyi parametreler:", best_params)
            svm_model = train_svm_with_best_params(X_train, y_train, best_params)
            
            # Test verisi üzerinde modeli değerlendirme
            y_pred = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Test verisi doğruluk oranı:", accuracy)
            
            # Modeli değerlendirme
            evaluate_model(y_test, y_pred)

        elif selected_model == "Naïve Bayes":
            # Naïve Bayes modelini eğitme
            st.subheader("Naïve Bayes Modeli")
            nb_model = train_naive_bayes(X_train, y_train)
            
            # Test verisi üzerinde modeli değerlendirme
            y_pred = nb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Test verisi doğruluk oranı:", accuracy)
            
            # Modeli değerlendirme
            evaluate_model(y_test, y_pred)

#//////////////////////////////////////////////////////

# Streamlit uygulaması
if __name__ == "__main__":
    main()


