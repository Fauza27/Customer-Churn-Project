import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
data = pd.read_csv('customer_churn.csv')

# Cleaning Data
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.loc[(data['TotalCharges'].isnull()) & (data['tenure'] == 0), 'TotalCharges'] = 0.0

# Memuat model Machine Learning dan object preprocessing
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('imputer_numeric.pkl', 'rb') as file:
    imputer_numeric = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('imputer_categorical.pkl', 'rb') as file:
    imputer_categorical = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Fungsi untuk melakukan preprocessing data input
def preprocess_input(input_data, imputer_numeric, scaler, imputer_categorical, encoder):
    # Mengonversi data input menjadi DataFrame
    df = pd.DataFrame([input_data])

    # Fitur rekayasa
    df['CLV'] = df['tenure'] * df['MonthlyCharges']
    df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure']
    df['AvgMonthlyCharges'].fillna(df['MonthlyCharges'], inplace=True)

    # Preprocessing fitur numerik dan kategorikal
    numerical_vars = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'AvgMonthlyCharges']
    categorical_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    X_numeric = imputer_numeric.transform(df[numerical_vars])
    X_numeric = scaler.transform(X_numeric)

    X_categorical = imputer_categorical.transform(df[categorical_vars])
    X_categorical = encoder.transform(X_categorical)


    X_preprocessed = np.hstack((X_numeric, X_categorical))
    
    return X_preprocessed



# Sidebar untuk memilih jenis analisis
st.sidebar.title('Churn Prediction Analysis')
analysis_type = st.sidebar.selectbox('Choose Analysis', 
                                     ['Churn Analysis', 'Model Testing'])

# Latar Belakang
if analysis_type == 'Churn Analysis':

    st.title('Churn Prediction Analysis')
    st.markdown("""
    ## Latar Belakang

    ### 1. Pendahuluan
    Dalam dunia bisnis saat ini, mempertahankan pelanggan yang sudah ada sering kali lebih penting dan lebih ekonomis daripada menarik pelanggan baru. 
    Salah satu tantangan utama yang dihadapi oleh perusahaan, terutama dalam industri yang kompetitif seperti telekomunikasi, keuangan, dan e-commerce, adalah churn pelanggan. 
    Churn pelanggan terjadi ketika pelanggan berhenti menggunakan produk atau layanan perusahaan dalam periode waktu tertentu.

    ### 2. Pentingnya Menangani Churn
    Menangani churn pelanggan sangat penting karena beberapa alasan:
    - Biaya Akuisisi Pelanggan Baru: Mendapatkan pelanggan baru umumnya lebih mahal daripada mempertahankan yang sudah ada. 
    Biaya pemasaran, promosi, dan insentif sering kali cukup tinggi.
    - Pendapatan Berkelanjutan: Pelanggan yang sudah ada biasanya memberikan aliran pendapatan yang stabil. 
    Kehilangan mereka berarti kehilangan sumber pendapatan berkelanjutan.
    - Reputasi dan Citra Merek: Tingkat churn yang tinggi dapat mempengaruhi citra merek dan persepsi pasar terhadap perusahaan. 
    Pelanggan yang loyal cenderung merekomendasikan perusahaan kepada orang lain, sedangkan churn dapat menimbulkan ulasan negatif.

    ### 3. Mengapa Menggunakan Data Science untuk Prediksi Churn
    Pendekatan tradisional untuk mengidentifikasi churn sering kali bersifat reaktif, seperti menunggu hingga pelanggan menghubungi dukungan untuk membatalkan layanan. 
    Dengan kemajuan dalam bidang data science dan machine learning, kini kita dapat memprediksi churn secara proaktif. Hal ini memungkinkan perusahaan untuk:
    - Mengambil Tindakan Proaktif: Dengan memprediksi pelanggan yang kemungkinan besar akan churn, perusahaan dapat mengambil tindakan preventif seperti menawarkan insentif atau meningkatkan layanan.
    - Personalisasi Layanan: Memahami faktor-faktor yang menyebabkan churn memungkinkan perusahaan untuk mempersonalisasi layanan sesuai dengan kebutuhan dan preferensi pelanggan.
    - Efisiensi Operasional: Dengan fokus pada pelanggan yang berisiko tinggi churn, perusahaan dapat mengalokasikan sumber daya secara lebih efektif.

    ### 4. Metodologi
    Proyek prediksi churn ini melibatkan penggunaan algoritma machine learning untuk mengidentifikasi pelanggan yang berisiko churn. Proses ini mencakup beberapa tahap:
    - Pengumpulan Data: Mengumpulkan data pelanggan yang mencakup informasi demografi, perilaku penggunaan, dan interaksi dengan perusahaan.
    - Eksplorasi dan Preprocessing Data: Memeriksa data untuk anomali, menangani nilai yang hilang, dan melakukan normalisasi data.
    - Pemilihan Fitur: Mengidentifikasi fitur-fitur yang paling relevan untuk prediksi churn, seperti lama penggunaan layanan, frekuensi transaksi, dan interaksi dengan dukungan pelanggan.
    - Pengembangan Model: Menggunakan algoritma machine learning seperti decision tree, random forest, atau logistic regression untuk membangun model prediksi.
    - Evaluasi Model: Mengevaluasi kinerja model menggunakan metrik seperti akurasi, precision, recall, AUC-ROC, dan Balanced Accuracy Score.
    - Implementasi dan Pemantauan: Mengimplementasikan model dalam sistem operasional perusahaan dan memantau kinerjanya secara terus-menerus untuk melakukan penyesuaian jika diperlukan.

    ### 5. Kesimpulan
    Memprediksi churn pelanggan adalah langkah strategis yang dapat memberikan keuntungan kompetitif bagi perusahaan. Dengan memanfaatkan teknik data science dan machine learning, perusahaan dapat meningkatkan retensi pelanggan, mengurangi biaya operasional, dan meningkatkan kepuasan pelanggan. Proyek ini bertujuan untuk mengembangkan model prediksi churn yang andal dan dapat diimplementasikan secara efektif dalam konteks bisnis.
    """)

    #Business Understanding
    st.header('Business Understanding')
    st.write("""
             Berdasarkan penjelasan yang telah diuraikan sebelumnya munculah pertanyaan penting, bagaimana perusahaan dapat secara efektif mengidentifikasi pelanggan yang berisiko churn dan mengambil langkah-langkah proaktif untuk mengurangi churn tersebut? Dengan memahami faktor-faktor yang mendorong pelanggan untuk berhenti menggunakan layanan, perusahaan dapat merancang strategi yang lebih baik untuk meningkatkan retensi pelanggan dan memastikan pertumbuhan bisnis yang berkelanjutan.
             """)
    
    #Problem Statement
    st.header('Problem Statements')
    st.write("""
            Untuk mencapai tujuan ini, proyek churn prediction ini difokuskan pada menjawab beberapa pertanyaan kunci:
             
            1. Berapa persentase pelanggan yang churn dan tidak churn?
            Mengetahui persentase pelanggan yang churn adalah langkah awal untuk memahami skala masalah dan menetapkan target retensi yang realistis.
            
            2. Apakah gender, SeniorCitizen, Partner, dan Dependents memiliki pengaruh terhadap churn?
            Mengidentifikasi apakah karakteristik demografis seperti gender, status sebagai senior citizen, memiliki pasangan, atau memiliki tanggungan mempengaruhi keputusan pelanggan untuk churn dapat membantu dalam segmentasi pelanggan dan pengembangan strategi retensi yang spesifik.
            
            3. Bagaimana distribusi tenure (masa berlangganan) antara pelanggan yang churn dan tidak churn?
            Memahami bagaimana lama masa berlangganan mempengaruhi churn dapat memberikan wawasan tentang kapan pelanggan paling berisiko churn dan memungkinkan perusahaan untuk mengambil tindakan pada waktu yang tepat.
            
            4. Apakah jenis layanan telepon (PhoneService, MultipleLines) dan layanan internet (InternetService, OnlineSecurity, OnlineBackup, 
            DeviceProtection, TechSupport, StreamingTV, StreamingMovies) berhubungan dengan churn?
            Menganalisis hubungan antara jenis layanan yang digunakan pelanggan dan churn dapat membantu dalam mengidentifikasi layanan mana yang perlu ditingkatkan atau dipromosikan untuk mengurangi churn.
            
            5. Bagaimana hubungan antara jenis kontrak (Contract) dan metode pembayaran (PaymentMethod) dengan churn?
            Memahami bagaimana jenis kontrak (bulanan, tahunan, dll.) dan metode pembayaran (kartu kredit, transfer bank, dll.) mempengaruhi churn dapat membantu dalam merancang penawaran kontrak dan metode pembayaran yang lebih efektif.
            
            6. Apakah ada perbedaan signifikan dalam MonthlyCharges dan TotalCharges antara pelanggan yang churn dan tidak churn?
            Menganalisis perbedaan biaya bulanan dan total biaya yang dibayarkan oleh pelanggan dapat memberikan wawasan tentang apakah harga layanan berperan dalam keputusan churn.
            
            7. Apakah terdapat korelasi antara fitur numerik seperti MonthlyCharges, TotalCharges, dan tenure dengan churn?
            Mengetahui korelasi antara fitur-fitur numerik dan churn dapat membantu dalam pengembangan model prediksi yang lebih akurat.
            
            8. Apakah ada interaksi menarik antara dua atau lebih fitur yang mempengaruhi churn? Misalnya, interaksi antara jenis kontrak dan metode pembayaran.
            Mengidentifikasi interaksi antara fitur-fitur dapat mengungkapkan pola-pola yang kompleks dan membantu dalam pengembangan strategi retensi yang lebih efektif.
        """)

    #Goals
    st.header('Goals')
    st.write("""
            Untuk menjawab pertanyaan tersebut, akan dilakukan analisis mendalam dengan tujuan sebagai berikut:
            1. Mengembangkan Model Prediksi Churn: Menggunakan teknik machine learning untuk membangun model yang dapat memprediksi pelanggan mana yang kemungkinan besar akan churn, berdasarkan data historis dan fitur-fitur yang relevan.
            2. Mengidentifikasi Faktor-faktor Penyebab Churn: Menentukan faktor-faktor kunci yang berkontribusi pada churn, sehingga perusahaan dapat fokus pada aspek-aspek yang paling mempengaruhi keputusan pelanggan untuk meninggalkan layanan.
            3. Merancang Strategi Retensi: Menggunakan hasil analisis untuk mengembangkan strategi retensi yang lebih efektif, termasuk personalisasi penawaran, peningkatan layanan, dan penyesuaian harga atau kontrak.
            4. Meningkatkan Efisiensi Operasional: Mengoptimalkan alokasi sumber daya dengan fokus pada pelanggan yang berisiko tinggi, sehingga perusahaan dapat mengurangi biaya operasional yang terkait dengan churn.

            Dengan pendekatan ini, proyek churn prediction bertujuan untuk memberikan perusahaan alat yang kuat untuk mempertahankan pelanggan, meningkatkan pendapatan, dan memperkuat posisi kompetitif di pasar.
             """)

    # Informasi Dataset

    st.title('Dataset Information')
    st.write("""
    Dataset yang digunakan pada project kali ini diambil dari platform kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)             
""")
    st.write(data.info())
    st.write("""
             Mengecek data type pada masing masing fitur
             """)
    st.code("""
            <class 'pandas.core.frame.DataFrame'>
            RangeIndex: 7043 entries, 0 to 7042
            Data columns (total 21 columns):
            #   Column            Non-Null Count  Dtype  
            ---  ------            --------------  -----  
            0   customerID        7043 non-null   object 
            1   gender            7043 non-null   object 
            2   SeniorCitizen     7043 non-null   int64  
            3   Partner           7043 non-null   object 
            4   Dependents        7043 non-null   object 
            5   tenure            7043 non-null   int64  
            6   PhoneService      7043 non-null   object 
            7   MultipleLines     7043 non-null   object 
            8   InternetService   7043 non-null   object 
            9   OnlineSecurity    7043 non-null   object 
            10  OnlineBackup      7043 non-null   object 
            11  DeviceProtection  7043 non-null   object 
            12  TechSupport       7043 non-null   object 
            13  StreamingTV       7043 non-null   object 
            14  StreamingMovies   7043 non-null   object 
            15  Contract          7043 non-null   object 
            16  PaperlessBilling  7043 non-null   object 
            17  PaymentMethod     7043 non-null   object 
            18  MonthlyCharges    7043 non-null   float64
            19  TotalCharges      7043 non-null   object 
            20  Churn             7043 non-null   object 
            dtypes: float64(1), int64(2), object(18)
            memory usage: 1.1+ MB
             """, language='python')
    st.write("""
             Pada dataframe data memiliki 21 feature atau column dan 7043 nilai non-null dalam setiap feature-feature yang ada. feature yang memiliki tipe data int64 sebanyak 2 sedangkan dengan feature yang memiliki tipe data float64 sebanyak 1. feature totalcharge memiliki tipe data yang salah sehingga harus diubah, dan sisanya merupakan feature yang bertipe data kategorikal.
             """)
    st.write("""
             Mengecek Missing Value pada dataset
             """)
    st.code("""
            data.isnull().sum()
            """, language='python')
    st.write("Jumlah Missing value data pada dataframe:", data.isnull().sum())
    st.write("""
             Mengecek data duplicate pada dataset
             """)
    st.code("""
            data.duplicated().sum()
            """, language='python')
    st.write("Jumlah duplikasi data pada dataframe:", data.duplicated().sum())
    st.write("""
             Melihat deskripsi statistik pada dataset
             """)
    st.write(data.describe(include='all'))

    st.subheader('Cleaning datatype')
    st.write("""
             fixing datatype
             """)
    st.code("""
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            """, language='python')
    st.write("""
             Mengecek kembali missing value
             """)
    st.code("""
            customerID           0
            gender               0
            SeniorCitizen        0
            Partner              0
            Dependents           0
            tenure               0
            PhoneService         0
            MultipleLines        0
            InternetService      0
            OnlineSecurity       0
            OnlineBackup         0
            DeviceProtection     0
            TechSupport          0
            StreamingTV          0
            StreamingMovies      0
            Contract             0
            PaperlessBilling     0
            PaymentMethod        0
            MonthlyCharges       0
            TotalCharges        11
            Churn                0
            """, language='python')
    st.write("""
             terdapat 11 missing value pada feature Total Charge, kita dapat mengetahui bahwa total charge merupakan perkalian dari feature tenure dengan monthly charge, jadi kita bisa mengetahui bahwa kemungkinan baris total charge yang kosong disebabkan oleh tenure yang bernilai 0
             """)
    st.code("""
            data.loc[(data['TotalCharges'].isnull()) & (data['tenure'] == 0), 'TotalCharges'] = 0.0
            """, language='python')
    st.write(data.isnull().sum())

    # Eksplorasi Data (EDA)

    st.title('Exploratory Data Analysis (EDA)')
    st.subheader('Persentase Pelanggan yang Churn dan Tidak Churn')
    
    churn_counts = data['Churn'].value_counts()
    churn_percent = churn_counts / len(data) * 100
    
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=140, colors=['blue', 'darkred'])
    ax1.set_title('Distribusi Churn')
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots()
    sns.barplot(x=churn_counts.index, y=churn_percent, palette=['blue', 'red'], ax=ax2)
    ax2.set_title('Persentase Pelanggan yang Churn dan Tidak Churn')
    ax2.set_ylabel('Persentase')
    st.pyplot(fig2)

    st.write("""
            - Grafik ini menunjukkan jumlah pelanggan yang churn dan tidak churn.
            - Dari grafik ini, terlihat bahwa mayoritas pelanggan tidak churn (sekitar 75%), sementara sekitar 25% pelanggan churn.
    """)
    
    st.subheader('Pengaruh Gender, SeniorCitizen, Partner, dan Dependents terhadap Churn')
    demographics = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    
    for feature in demographics:
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=feature, hue='Churn', palette='bright', ax=ax)
        ax.set_title(f'Churn Berdasarkan {feature}')
        st.pyplot(fig)

    st.write("""
            - Grafik ini menunjukkan distribusi churn berdasarkan gender, status SeniorCitizen (warga senior), status partner, dan status dependents (tanggungan).
            - Tampak bahwa baik laki-laki maupun perempuan memiliki pola churn yang serupa, dengan lebih banyak pelanggan yang tidak churn dibandingkan yang churn dalam kedua kelompok gender.
            - Pelanggan yang bukan warga senior memiliki angka churn yang lebih tinggi dibandingkan warga senior. Namun, ada lebih banyak pelanggan non-senior secara keseluruhan.
            - Pelanggan yang memiliki partner cenderung memiliki churn yang lebih rendah dibandingkan pelanggan yang tidak memiliki partner.
            - Pelanggan tanpa tanggungan memiliki angka churn yang lebih tinggi dibandingkan pelanggan dengan tanggungan.
             
    """)
    
    st.subheader('Distribusi Tenure antara Pelanggan yang Churn dan Tidak Churn')
    
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['CLV'] = data['tenure'] * data['MonthlyCharges']
    
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='CLV', y='MonthlyCharges', hue='Churn', data=data, ax=ax3)
    ax3.set_title('Customer Lifetime Value vs Monthly Charges')
    st.pyplot(fig3)
    
    data['TenureBin'] = pd.cut(data['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], labels=['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years'])
    churn_by_contract_tenure = data.groupby(['Contract', 'TenureBin'])['Churn'].mean().unstack()
    
    fig4, ax4 = plt.subplots()
    churn_by_contract_tenure.plot(kind='bar', stacked=False, ax=ax4)
    ax4.set_title('Churn Rate by Contract Type and Tenure')
    ax4.set_xlabel('Contract Type')
    ax4.set_ylabel('Churn Rate')
    st.pyplot(fig4)
    
    data['Churn'] = data['Churn'].map({1: 'Yes', 0: 'No'})

    st.write("""
            - Grafik ini menunjukkan tingkat churn berdasarkan jenis kontrak dan masa berlangganan (tenure).
            - Pelanggan dengan kontrak bulanan memiliki tingkat churn tertinggi, terutama pada masa berlangganan 0-1 tahun.
            - Pelanggan dengan kontrak satu atau dua tahun memiliki tingkat churn yang jauh lebih rendah, menunjukkan bahwa kontrak jangka panjang dapat mengurangi churn.
            - Terlihat bahwa ada banyak variasi dalam Monthly Charges di seluruh CLV.
            - Pelanggan yang churn (ditandai dengan warna oranye) cenderung tersebar di berbagai nilai CLV dan Monthly Charges, meskipun ada kecenderungan untuk churn pada tingkat Monthly Charges yang lebih tinggi.
    """)
    
    st.subheader('Analisis Layanan Telepon dan Internet terhadap Churn')
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for service in services:
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=service, hue='Churn', palette='bright', ax=ax)
        ax.set_title(f'Churn Berdasarkan {service}')
        st.pyplot(fig)
    

    st.write("""
            - Pelanggan yang tidak memiliki layanan telepon cenderung memiliki churn yang lebih rendah.
            - Pelanggan yang memiliki layanan telepon memiliki angka churn yang lebih tinggi, tetapi jumlah pelanggan yang memiliki layanan telepon jauh lebih banyak daripada yang tidak.
            - Pelanggan yang memiliki beberapa saluran telepon cenderung memiliki tingkat churn yang lebih tinggi dibandingkan dengan pelanggan yang tidak memiliki beberapa saluran telepon.
            - Pelanggan yang tidak memiliki layanan telepon memiliki tingkat churn yang paling rendah.
            - Pelanggan yang menggunakan layanan internet fiber optic memiliki tingkat churn yang lebih tinggi dibandingkan dengan pelanggan yang menggunakan DSL atau tidak memiliki layanan internet.
            - Pelanggan yang tidak memiliki layanan internet memiliki tingkat churn yang paling rendah.
            - Pelanggan yang tidak memiliki layanan keamanan online cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki layanan keamanan online atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
            - Pelanggan yang tidak memiliki layanan backup online cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki layanan backup online atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
            - Pelanggan yang tidak memiliki perlindungan perangkat cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki perlindungan perangkat atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
            - Pelanggan yang tidak memiliki dukungan teknis cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki dukungan teknis atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
            - Pelanggan yang tidak memiliki layanan streaming TV cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki layanan streaming TV atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
            - Pelanggan yang tidak memiliki layanan streaming film cenderung memiliki tingkat churn yang lebih tinggi.
            - Pelanggan yang memiliki layanan streaming film atau tidak memiliki layanan internet memiliki tingkat churn yang lebih rendah.
    """)

    st.subheader('Analisis Jenis Kontrak dan Metode Pembayaran terhadap Churn')
    contract_payment = ['Contract', 'PaymentMethod']
    
    for feature in contract_payment:
        fig, ax = plt.subplots()
        sns.countplot(data=data, x=feature, hue='Churn', palette='bright', ax=ax)
        ax.set_title(f'Churn Berdasarkan {feature}')
        st.pyplot(fig)
    
    st.write("""
            - Pelanggan dengan kontrak bulanan memiliki tingkat churn yang jauh lebih tinggi dibandingkan dengan pelanggan yang memiliki kontrak satu tahun atau dua tahun.
            - Pelanggan dengan kontrak dua tahun memiliki tingkat churn yang paling rendah, diikuti oleh kontrak satu tahun.
            - Pelanggan yang menggunakan cek elektronik memiliki tingkat churn yang lebih tinggi dibandingkan dengan metode pembayaran lainnya.
            - Pelanggan yang menggunakan transfer bank otomatis atau kartu kredit memiliki tingkat churn yang lebih rendah.
    """)
    
    st.subheader('Analisis Perbedaan MonthlyCharges dan TotalCharges antara Pelanggan yang Churn dan Tidak Churn')
    charges = ['MonthlyCharges', 'TotalCharges']
    
    for charge in charges:
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='Churn', y=charge, palette='bright', ax=ax)
        ax.set_title(f'{charge} Berdasarkan Churn')
        st.pyplot(fig)
    
    st.write("""
            - Grafik ini menunjukkan distribusi MonthlyCharges (biaya bulanan) berdasarkan status churn.
            - Pelanggan yang churn memiliki median MonthlyCharges yang lebih tinggi dibandingkan pelanggan yang tidak churn.
            - Distribusi MonthlyCharges untuk pelanggan yang churn lebih tersebar, menunjukkan bahwa ada variasi yang lebih besar dalam biaya bulanan mereka.
            - Grafik ini menunjukkan distribusi TotalCharges (biaya total) berdasarkan status churn.
            - Pelanggan yang churn memiliki median TotalCharges yang lebih rendah dibandingkan pelanggan yang tidak churn.
            - Ini mungkin menunjukkan bahwa pelanggan yang churn cenderung berlangganan untuk jangka waktu yang lebih pendek, sehingga akumulasi biaya total mereka lebih rendah.
    """)

    st.subheader('Analisis Korelasi antara Fitur Numerik dengan Churn')
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    correlation = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'Churn']].corr()
    
    fig5, ax5 = plt.subplots()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title('Korelasi antara Fitur Numerik dengan Churn')
    st.pyplot(fig5)
    
    data['Churn'] = data['Churn'].map({1: 'Yes', 0: 'No'})

    st.write("""
            - Grafik ini menunjukkan korelasi antara fitur numerik seperti tenure, MonthlyCharges, TotalCharges, dan churn.
            - Korelasi antara tenure dan churn adalah -0.35, menunjukkan bahwa semakin lama pelanggan berlangganan, semakin kecil kemungkinan mereka untuk churn.
            - MonthlyCharges dan churn memiliki korelasi positif sebesar 0.19, menunjukkan bahwa biaya bulanan yang lebih tinggi sedikit berkorelasi dengan peningkatan kemungkinan churn.
            - TotalCharges memiliki korelasi negatif dengan churn (-0.20), menunjukkan bahwa biaya total yang lebih tinggi sedikit berkorelasi dengan penurunan kemungkinan churn.
            - Korelasi antara TotalCharges dan tenure sangat tinggi (0.83), yang diharapkan karena biaya total cenderung meningkat seiring dengan lamanya masa berlangganan.
            - Korelasi antara MonthlyCharges dan TotalCharges adalah 0.65, yang menunjukkan bahwa biaya bulanan yang lebih tinggi berkontribusi secara signifikan terhadap biaya total.
             """)
    
    # Kesimpulan
    st.title('Kesimpulan')
    st.markdown("""
    
    Berdasarkan analisis data churn ini, beberapa kesimpulan yang dapat diambil adalah:
    
    1. **Distribusi Churn**: Persentase pelanggan yang churn cukup signifikan, menunjukkan pentingnya langkah-langkah preventif untuk mengurangi churn.
    
    2. **Faktor Demografi**: Gender dan status sebagai SeniorCitizen, Partner, dan Dependents memiliki pengaruh yang berbeda-beda terhadap churn.
    
    3. **Pengaruh Tenure**: Pelanggan dengan tenure yang lebih pendek cenderung lebih mungkin untuk churn. Tenure yang lebih panjang menunjukkan loyalitas yang lebih besar.
    
    4. **Layanan yang Digunakan**: Layanan seperti PhoneService, InternetService, dan layanan tambahan lainnya memiliki korelasi dengan churn. Pelanggan yang menggunakan lebih banyak layanan cenderung lebih loyal.
    
    5. **Jenis Kontrak dan Metode Pembayaran**: Jenis kontrak dan metode pembayaran juga mempengaruhi churn. Pelanggan dengan kontrak bulanan lebih mungkin churn dibandingkan dengan pelanggan dengan kontrak jangka panjang.
    
    6. **Biaya Bulanan dan Total**: Pelanggan dengan biaya bulanan dan total yang lebih tinggi cenderung memiliki tingkat churn yang lebih rendah, mungkin karena mereka merasa lebih terikat dengan layanan yang mereka bayar lebih mahal.
    
    7. **Korelasi antara Fitur Numerik**: Analisis korelasi menunjukkan hubungan antara beberapa fitur numerik dengan churn, seperti tenure, MonthlyCharges, dan TotalCharges.
    
    Secara keseluruhan, analisis ini memberikan wawasan penting tentang faktor-faktor yang mempengaruhi churn dan dapat digunakan sebagai dasar untuk mengembangkan strategi retensi pelanggan yang lebih efektif.
    """)

    #Recomendation Action
    st.title('Rekomendation Action')
    st.markdown("""
                1. Meningkatkan Loyalitas Pelanggan dengan Kontrak Jangka Panjang:
                - Promosikan kontrak satu atau dua tahun dengan insentif khusus untuk mengurangi churn pelanggan dengan kontrak bulanan.

                2. Optimalkan Metode Pembayaran:
                - Mendorong pelanggan untuk menggunakan metode pembayaran otomatis seperti transfer bank atau kartu kredit dengan menawarkan diskon atau insentif.

                3. Fokus pada Layanan Tambahan:
                - Tawarkan layanan tambahan seperti OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, dan StreamingMovies dengan harga paket yang lebih murah atau sebagai bagian dari paket langganan untuk meningkatkan retensi.

                4. Kampanye Khusus untuk SeniorCitizen:
                - Buat program loyalitas atau layanan khusus untuk SeniorCitizen untuk mengurangi tingkat churn di kelompok ini.

                5. Pendekatan Khusus untuk Pelanggan Tanpa Partner atau Dependents:
                - Buat kampanye khusus atau layanan tambahan yang menarik bagi pelanggan yang tidak memiliki partner atau dependents untuk meningkatkan retensi.

                6. Pengelolaan Biaya Bulanan:
                - Monitor pelanggan dengan MonthlyCharges tinggi dan berikan penawaran khusus atau insentif untuk memastikan mereka tetap berlangganan.

                7. Peningkatan Kualitas Layanan Fiber Optic:
                - Pelanggan fiber optic menunjukkan tingkat churn yang lebih tinggi. Evaluasi dan tingkatkan kualitas layanan fiber optic untuk mengurangi churn di segmen ini.

                Dengan tindakan-tindakan ini, perusahaan dapat mengurangi churn dan meningkatkan loyalitas pelanggan.
                """)

    st.write("""
             Sekarang akan dilakukan proses pembuatan model machine learning untuk memprediksi apakah suatu customer akan melakukan churn atau tidak.
             """)

    #Feature Engineering
    st.header("Feature Preprocessing")
    st.code("""
            #menentukan fitur numerik dan kategorik

            numerical_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            from imblearn.over_sampling import SMOTE

            #Membagi dataset menjadi data training dan data testing

            X = data.drop('Churn', axis = 1)
            Y = data['Churn']

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            #Preprocessing untuk fitur numerik

            imputer_numeric = SimpleImputer(strategy='median')
            scaler = StandardScaler()

            X_train_numeric = imputer_numeric.fit_transform(X_train[numerical_vars])
            X_test_numeric = imputer_numeric.transform(X_test[numerical_vars])

            X_train_numeric = scaler.fit_transform(X_train_numeric)
            X_test_numeric = scaler.transform(X_test_numeric)

            #Preprocessing untuk fitur kategorikal

            imputer_categorical = SimpleImputer(strategy='constant', fill_value='missing')
            encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')

            X_train_categorical = imputer_categorical.fit_transform(X_train[categorical_vars])
            X_test_categorical = imputer_categorical.transform(X_test[categorical_vars])

            X_train_categorical = encoder.fit_transform(X_train_categorical)
            X_test_categorical = encoder.transform(X_test_categorical)

            Y_train = Y_train.map({'Yes': 1, 'No': 0})
            Y_test = Y_test.map({'Yes': 1, 'No': 0})

            X_train_preprocessed = np.hstack((X_train_numeric, X_train_categorical))
            X_test_preprocessed = np.hstack((X_test_numeric, X_test_categorical))

            smote = SMOTE(random_state=42)
            X_train_smote, Y_train_smote = smote.fit_resample(X_train_preprocessed, Y_train)

            cat_feature_names = encoder.get_feature_names_out(categorical_vars)
            feature_names = numerical_vars + list(cat_feature_names)

            X_train_smote = pd.DataFrame(X_train_smote, columns=feature_names)
            X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names)
            """)
    
    #Feature Engineering
    st.header('Feature Engineering')
    st.code("""
            #Membuat fitur baru yaitu customer life time yang dimana merupakan perkalian antara fitur tenure dan monthlycharge, dan fitu average monthl charge yang dimana merupakan pembagian antara fitur totalcharge dan tenure

            def features_engineer(data):
                data['CLV'] = data['tenure'] * data['MonthlyCharges']
                data['AvgMonthlyCharges'] = data['TotalCharges'] / data['tenure']
                data['AvgMonthlyCharges'].fillna(data['MonthlyCharges'], inplace=True)
                return data

            X_train_smote = features_engineer(X_train_smote)
            X_test_preprocessed = features_engineer(X_test_preprocessed)

            new_numeric_features = ['CLV', 'AvgMonthlyCharges']
            scaler = StandardScaler()

            X_train_smote[new_numeric_features] = scaler.fit_transform(X_train_smote[new_numeric_features])
            X_test_preprocessed[new_numeric_features] = scaler.transform(X_test_preprocessed[new_numeric_features])

            X_train_smote.head()
            """)

    #Membangun Model
    st.header("Pembangunan Model")
    st.write("""
    Pada tahap ini, lima algoritma machine learning digunakan untuk membangun model prediksi churn pelanggan:

    - **Logistic Regression**: Dipilih karena kesederhanaan dan kemampuannya dalam memberikan baseline model yang mudah diinterpretasikan. Logistic Regression bekerja dengan baik pada dataset besar dan mampu menangani multikolinearitas, meskipun rentan terhadap outlier dan mengasumsikan hubungan linear antara fitur dan probabilitas churn.

    - **Decision Tree**: Dipilih karena kemampuannya menghasilkan model yang mudah diinterpretasikan tanpa banyak prapemrosesan data. Namun, algoritma ini cenderung overfitting dan kurang stabil.

    - **Random Forest**: Dipilih karena kemampuannya dalam mengurangi overfitting dan memberikan akurasi yang tinggi. Random Forest juga robust terhadap missing values, meski kurang interpretatif dan membutuhkan lebih banyak sumber daya.

    - **Gradient Boosting**: Dipilih karena kemampuannya untuk mencapai akurasi tinggi dan robust terhadap outlier. Namun, waktu latihnya lebih lama dan parameter tuningnya cukup kompleks.

    - **Support Vector Machine (SVM)**: Dipilih karena efektivitasnya dalam menangani data dimensi tinggi dan fleksibilitas dalam menangani masalah non-linear melalui kernel trick. SVM memerlukan tuning parameter yang hati-hati dan kurang efektif pada dataset yang sangat besar.
    """)

    # Tampilkan kode untuk membangun model
    st.code("""
    # Import library
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.model_selection import cross_val_score

    # Membuat fungsi model
    def train_and_evaluate(model, X_train, X_test, Y_train, Y_test):
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        roc_auc = roc_auc_score(Y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(Y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'balanced_accuracy': balanced_acc,
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        }

    # Membangun model
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        results[name] = train_and_evaluate(model, X_train_smote, X_test_preprocessed, Y_train_smote, Y_test)
    """, language='python')

    # Evaluasi Model
    st.header("Evaluasi Model")
    st.write("""
    Setelah membangun model, setiap model dievaluasi menggunakan berbagai metrik seperti Accuracy, Precision, Recall, F1-Score, ROC AUC, dan Balanced Accuracy. Logistic Regression menunjukkan performa terbaik dalam hal recall dan balanced accuracy, yang penting untuk mendeteksi churn.
    """)

    # Tampilkan kode untuk evaluasi model
    st.code("""
    def evaluate_model(model, X, y, name):
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        print(f"Evaluation metrics for {name}:")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"Precision: {precision_score(y, y_pred):.4f}")
        print(f"Recall: {recall_score(y, y_pred):.4f}")
        print(f"F1-score: {f1_score(y, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y, y_pred_proba):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    for name, model in models.items():
        evaluate_model(model, X_test_preprocessed, Y_test, name)
    """, language='python')

    #Hasil Evaluasi untuk Logistic Regression
    st.header("Evaluation metrics for Logistic Regression:")
    st.write("""
    - Accuracy: 0.7516
    - Precision: 0.5193
    - Recall: 0.8284
    - F1-score: 0.6384
    - ROC AUC: 0.8601
    - Balanced Accuracy: 0.7762
    """)
    # Mengganti ini dengan plot confusion matrix Anda
    image_logistic = "\img\logistic.png"
    st.image(image_logistic, caption="Confusion Matrix - Logistic Regression")

    # Contoh Hasil Evaluasi untuk Decision Tree
    st.header("Evaluation metrics for Decision Tree:")
    st.write("""
    - Accuracy: 0.7402
    - Precision: 0.5086
    - Recall: 0.5576
    - F1-score: 0.5320
    - ROC AUC: 0.6825
    - Balanced Accuracy: 0.6818
    """)
    image_decision_tree = "img\decision tree.png"
    st.image(image_decision_tree, caption="Confusion Matrix - Decision Tree")

    # Contoh Hasil Evaluasi untuk Random Forest
    st.header("Evaluation metrics for Random Forest:")
    st.write("""
    - Accuracy: 0.7928
    - Precision: 0.6069
    - Recall: 0.6166
    - F1-score: 0.6117
    - ROC AUC: 0.8381
    - Balanced Accuracy: 0.7364
    """)
    image_random_forest = "img\Random forest.png"
    st.image(image_random_forest, caption="Confusion Matrix - Random Forest")

    # Contoh Hasil Evaluasi untuk Gradient Boosting
    st.header("Evaluation metrics for Gradient Boosting:")
    st.write("""
    - Accuracy: 0.7935
    - Precision: 0.5907
    - Recall: 0.7158
    - F1-score: 0.6473
    - ROC AUC: 0.8581
    - Balanced Accuracy: 0.7686
    """)
    image_gradient_boosting = "img\gradient boosting.png"
    st.image(image_gradient_boosting, caption="Confusion Matrix - Gradient Boosting")

    # Contoh Hasil Evaluasi untuk SVM
    st.header("Evaluation metrics for SVM:")
    st.write("""
    - Accuracy: 0.7630
    - Precision: 0.5360
    - Recall: 0.7775
    - F1-score: 0.6346
    - ROC AUC: 0.8473
    - Balanced Accuracy: 0.7676
    """)
    image_svm = "img\svm.png"
    st.image(image_svm, caption="Confusion Matrix - SVM")

    st.header("Hyperparamter Tunning")
    st.write("""
    Saya melakukan hyperparameter tuning untuk meningkatkan performa model, khususnya pada Logistic Regression dan Gradient Boosting:
    """)

    # Tampilkan kode untuk membangun model
    st.code("""
            ##Hyperparameter Tuning

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    ##melakukan hyperparameter tuning dengan grid search cv untuk model logistic regression untuk meningkatkan akurasi model

    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000]
    }

    log_reg = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid,
                            cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=2)

    grid_search.fit(X_train_smote, Y_train_smote)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best balanced accuracy score: ", grid_search.best_score_)

    best_log_reg = grid_search.best_estimator_
    best_log_reg.fit(X_train_smote, Y_train_smote)

    y_pred = best_log_reg.predict(X_test_preprocessed)
    y_pred_proba = best_log_reg.predict_proba(X_test_preprocessed)[:, 1]
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)
    balanced_accuracy = balanced_accuracy_score(Y_test, y_pred)

    print("Evaluation metrics for the tuned Logistic Regression model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuned Logistic Regression')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    ##melakukan hyperparameter tuning dengan random search cv untuk model logistic regression untuk meningkatkan akurasi model

    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import uniform, loguniform

    # Define the parameter distribution for Logistic Regression
    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': loguniform(1e-4, 1e2),
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000]
    }

    log_reg = LogisticRegression(random_state=42)

    # Instantiate the randomized search model
    random_search = RandomizedSearchCV(estimator=log_reg, param_distributions=param_dist,
                                    n_iter=100, cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=2, random_state=42)


    random_search.fit(X_train_smote, Y_train_smote)

    print("Best parameters found: ", random_search.best_params_)
    print("Best balanced accuracy score: ", random_search.best_score_)

    best_log_reg = random_search.best_estimator_
    best_log_reg.fit(X_train_smote, Y_train_smote)


    y_pred = best_log_reg.predict(X_test_preprocessed)
    y_pred_proba = best_log_reg.predict_proba(X_test_preprocessed)[:, 1]
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)
    balanced_accuracy = balanced_accuracy_score(Y_test, y_pred)

    print("Evaluation metrics for the tuned Logistic Regression model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuned Logistic Regression')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    ##Karena tingkat akurasi model linear regression masih belum memuaskan, maka akan dicoba lakukan hyperparameter tuning dengan grid search cv untuk model gradient boosting yang merupakan model dengan tingkat akurasi kedua setelah logistic regression untuk meningkatkan akurasi model

    # Define the parameter grid for Gradient Boosting
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a base model
    gb = GradientBoostingClassifier(random_state=42)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid,
                            cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train_smote, Y_train_smote)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best balanced accuracy score: ", grid_search.best_score_)

    # Train the best model on the entire training set
    best_gb = grid_search.best_estimator_
    best_gb.fit(X_train_smote, Y_train_smote)

    # Evaluate the best model on the test set
    y_pred = best_gb.predict(X_test_preprocessed)
    y_pred_proba = best_gb.predict_proba(X_test_preprocessed)[:, 1]
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)
    balanced_accuracy = balanced_accuracy_score(Y_test, y_pred)

    print("Evaluation metrics for the tuned Gradient Boosting model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuned Gradient Boosting')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    """, language='python')

    st.write("""
            Kesimpulan
            - Recall dan Balanced Accuracy: LR tanpa tuning memiliki nilai terbaik pada recall dan balanced accuracy.
            - Precision: GBC memiliki precision yang terbaik
            - F1-score dan ROC AUC: LR tanpa tuning sedikit lebih baik daripada GBC dan LR dengan tuning dalam hal F1-score dan ROC AUC.

            Karena model logistic regression tanpa tunning memiliki kemampuan terbaik dalam mendeteksi kasus positif (recall) dan memiliki distribusi performa yang seimbang antara kelas positif dan negatif, maka model yang akan dipilih pada kasus dataset ini adalah logistic regression        
    """)


#Melakukan prediksi terhadap model yang telah dibuat
elif analysis_type == 'Model Testing':
    # Membuat input form di Streamlit
    st.title("Model Testing")
    st.write("""
             jawablah pertanyaan-pertanyaan berikut ini untuk memprediksi apakah suatu customer akan melakukan churn atau tidak
             """)
    st.header("Masukkan data pelanggan:")

    gender = st.selectbox("Masukkan gender Anda:", ['Female', 'Male'])
    SeniorCitizen = st.selectbox("Apakah Anda termasuk Senior Citizen?", [0, 1])
    Partner = st.selectbox("Apakah Anda punya partner?", ['Yes', 'No'])
    Dependents = st.selectbox("Apakah Anda memiliki tanggungan?", ['Yes', 'No'])
    tenure = st.number_input("Berapa lama Anda menjadi pelanggan (tenure)?", min_value=0, max_value=100)
    PhoneService = st.selectbox("Apakah Anda menggunakan layanan telepon?", ['Yes', 'No'])
    MultipleLines = st.selectbox("Apakah Anda memiliki beberapa jalur telepon?", ['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Jenis layanan internet yang Anda gunakan:", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Apakah Anda memiliki keamanan online?", ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.selectbox("Apakah Anda memiliki backup online?", ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.selectbox("Apakah Anda memiliki perlindungan perangkat?", ['Yes', 'No', 'No internet service'])
    TechSupport = st.selectbox("Apakah Anda menggunakan layanan dukungan teknis?", ['Yes', 'No', 'No internet service'])
    StreamingTV = st.selectbox("Apakah Anda menggunakan layanan streaming TV?", ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.selectbox("Apakah Anda menggunakan layanan streaming film?", ['Yes', 'No', 'No internet service'])
    Contract = st.selectbox("Jenis kontrak Anda:", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Apakah Anda menggunakan tagihan tanpa kertas?", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Metode pembayaran yang Anda gunakan:", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.number_input("Berapa biaya bulanan Anda?", min_value=0.0)
    TotalCharges = st.number_input("Berapa total biaya Anda?", min_value=0.0)


    # Mengumpulkan input dalam dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Tombol untuk memulai prediksi
    if st.button("Prediksi"):
        # Preprocessing data input
        X_input = preprocess_input(input_data, imputer_numeric, scaler, imputer_categorical, encoder)
        
        # Melakukan prediksi
        prediction = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)[:, 1]

        # Menampilkan hasil prediksi
        if prediction[0] == 1:
            st.error(f"Pelanggan ini kemungkinan besar akan churn dengan probabilitas {prediction_proba[0]:.2f}.")
        else:
            st.success(f"Pelanggan ini kemungkinan besar tidak akan churn dengan probabilitas {1-prediction_proba[0]:.2f}.")

