from flask import Flask, render_template, request, jsonify,send_file,redirect, url_for, session,send_from_directory
import pandas as pd
import os
import warnings
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import base64
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import openpyxl
import pickle
import folium
from geopy.geocoders import Nominatim
import json
import csv
from folium import IFrame
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
import pyodbc
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine
from sqlalchemy import text
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
import joblib
import PyPDF2
import plotly
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objs as go
import random
import io
from io import BytesIO
import tempfile
from cryptography.fernet import Fernet


matplotlib.use('Agg')

app = Flask(__name__)

app.jinja_env.autoescape = True

app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=LAPTOP-6OET6DB6;Database=cv_project;Trusted_Connection=yes;'
db = SQLAlchemy(app)



class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)



app.secret_key = os.urandom(24)


# Son etkinlik zamanını saklamak için bir sözlük
last_activity = {}

# Giriş gerektiren sayfaların listesi
login_required_pages = ['aday_degerlendir','dashboard', 'ilan_ekle', 'acilan_ilanlar','gorsellestirme','Aday_basvuruları','basvurular', 'basvurulan_lokasyonlar','remove_position','register']

# Admin sayfalarının listesi
admin_pages = ['aday_degerlendir','dashboard', 'ilan_ekle', 'acilan_ilanlar','gorsellestirme','Aday_basvuruları','basvurular', 'basvurulan_lokasyonlar','remove_position','register']

# Kullanıcıların oturumlarını takip etmek için bir sözlük
user_sessions = {}

# Oturum süresi (saniye cinsinden)
SESSION_TIMEOUT = 1200


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']


        user = User.query.filter(text("username = :username AND password = :password")).params(username=username, password=password).first()

        # Eğer kullanıcı varsa ve şifre doğruysa oturumu başlat
        if user:
            session['user'] = username
            user_sessions[username] = datetime.now()
            return redirect(url_for('dashboard'))
        else:
            # Kullanıcı yoksa veya şifre uyuşmuyorsa hata mesajı gösterilir
            return render_template('login.html', message='Hatalı Şifre Yada Kullanıcı Adı')

    return render_template('login.html')



@app.route('/<page>')
def render_page(page):
    if 'user' not in session:
        return redirect(url_for('login'))

    if page in login_required_pages:
        username = session['user']
        user_sessions[username] = datetime.now()

    if page in admin_pages:
        if 'user' not in session:  # Kullanıcı girişi yapılmamışsa login sayfasına yönlendir
            return redirect(url_for('login'))
        else:
            username = session['user']
            user_sessions[username] = datetime.now()

    return render_template(page + '.html')

@app.route('/logout')
def logout():
    if 'user' in session:
        session.pop('user')
    return redirect(url_for('login'))

df_basvurular = None
df_sirket_pozisyonlari=None

@app.before_request
def before_request():
    if 'user' in session:
        username = session['user']
        if (datetime.now() - user_sessions.get(username, datetime.min)) > timedelta(seconds=SESSION_TIMEOUT):
            # Oturum süresi aşıldığında oturumu sonlandır
            session.pop('user')


def adaylar():
    global df_basvurular
    query = """
           SELECT 
               name AS Ad, 
               surname AS Soyad, 
               dob AS [Doğum Tarihi], 
               education AS [Eğitim Seviyesi], 
               position AS Pozisyon, 
               experience AS [İş Deneyimi (Yıl)], 
               selectedSkills AS Teknolojiler, 
               languages AS [Yabancı Dil], 
               languageslevel AS [Yabancı Dil Seviyesi], 
               intro AS [Aday Profili], 
               location AS İkametgah, 
               number AS [Telefon Numarası], 
               email AS Eposta 
           FROM basvurular
       """
    try:
        df_basvurular = pd.read_sql(query, db.engine)
        if df_basvurular.empty:
            df_basvurular = pd.DataFrame(
                columns=["Ad", "Soyad", "Doğum Tarihi", "Eğitim Seviyesi", "Pozisyon", "İş Deneyimi (Yıl)",
                         "Teknolojiler", "Yabancı Dil", "Yabancı Dil Seviyesi", "Aday Profili", "İkametgah",
                         "Telefon Numarası", "Eposta"])
    except Exception as e:
        print("Veritabanından veri çekilirken hata oluştu:", e)
        df_basvurular = pd.DataFrame(
            columns=["Ad", "Soyad", "Doğum Tarihi", "Eğitim Seviyesi", "Pozisyon", "İş Deneyimi (Yıl)", "Teknolojiler",
                     "Yabancı Dil", "Yabancı Dil Seviyesi", "Aday Profili", "İkametgah", "Telefon Numarası", "Eposta"])
    return df_basvurular

def sirket():
    global df_sirket_pozisyonlari
    sirket_pozisyonlari = SirketPozisyon.query.all()
    df_sirket_pozisyonlari = pd.DataFrame([(pozisyon.pozisyon, pozisyon.teknolojiler, pozisyon.yabanci_dil,
                                            pozisyon.yabanci_dil_seviyesi, pozisyon.aday_profil,
                                            pozisyon.egitim_seviyesi, pozisyon.calisma_sekli)
                                           for pozisyon in sirket_pozisyonlari],
                                          columns=["Pozisyon", "Teknolojiler", "Yabancı Dil",
                                                   "Yabancı Dil Seviyesi", "Aday Profili",
                                                   "Eğitim Seviyesi", "Çalışma Şekli"])
    return df_sirket_pozisyonlari

@app.route('/register', methods=['GET', 'POST'])  # Route'u güncelledik
def register():
    if 'user' in session:
        return render_template('register.html')

    return redirect(url_for('login'))

@app.route('/register_fonction', methods=['POST'])  # Yeni route
def register_post():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Yeni kullanıcı oluştur
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        # Kullanıcı oluşturulduktan sonra giriş sayfasına yönlendir
        return render_template('register.html', message='{} Yönetici Kaydı Başarıyla Eklendi'.format(username))

    # Eğer POST isteği değilse, kayıt sayfasına yönlendir
    return redirect(url_for('register'))


warnings.filterwarnings("ignore")


#nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))



def turkce_gramerleri_kaldir(metin):
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', ' ', metin)
    kelimeler = metin.split()
    temiz_metin = [kelime for kelime in kelimeler]
    return ' '.join(temiz_metin)

"""morphology = TurkishMorphology.create_with_defaults()
def turkce_gramerleri_kaldir(metin):
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', ' ', metin)
    kelimeler = metin.split()
    kelimeler = [kelime for kelime in kelimeler if kelime not in turkish_stopwords]
    temiz_metin = []
    for kelime in kelimeler:
        analysis = morphology.analyze(kelime)
        if analysis:
            lemmas = analysis[0].item.lemmas
            if lemmas:
                temiz_metin.append(lemmas[0])
            else:
                temiz_metin.append(kelime)
        else:
            temiz_metin.append(kelime)
    return ' '.join(temiz_metin)"""

def preprocess_data(df):
    df['Ön İşlenmiş Veri'] = df['Teknolojiler'] + " " + df['Yabancı Dil'] + " " + df['Yabancı Dil Seviyesi'] + " " + df["Eğitim Seviyesi"]
    df['Ön İşlenmiş Veri'] = df['Ön İşlenmiş Veri'].apply(turkce_gramerleri_kaldir)
    return df


@app.route('/evaluate', methods=['POST'])
def evaluate():
    global sonuç_df  # Global değişkene erişim sağlanıyor
    df_basvurular = adaylar()
    df_sirket_pozisyonlari = sirket()

    if df_basvurular.empty:
        return "Yapılan Bir Başvuru Bulunamadı",400

    if df_sirket_pozisyonlari.empty:
        return "Henüz Bir İlan Açılmadı",400
    df_sirket_pozisyonlari = preprocess_data(df_sirket_pozisyonlari)
    df_basvurular = preprocess_data(df_basvurular)

    tfidf_vectorizer = TfidfVectorizer()

    tüm_veri = df_sirket_pozisyonlari["Ön İşlenmiş Veri"].tolist() + df_basvurular["Ön İşlenmiş Veri"].tolist()

    if not tüm_veri:
        return "TF-IDF işlemi için yeterli veri yok.", 400

    tfidf_matrix = tfidf_vectorizer.fit_transform(tüm_veri)
    #df_basvurular.drop('Ön İşlenmiş Veri',axis=1, inplace=True)
    başvuran_aday_sayacı = range(len(df_sirket_pozisyonlari), len(tüm_veri))

    # Form verilerini al
    formData = request.form

    # Veri işlemlerini yap sos
    sonuç_df = pd.DataFrame(columns=["Başvuran Kişi","Email", "Başvurulan Pozisyon", "Pozisyonun Uygunluk Oranı %"])
    uygunluk_skoru=40
    for index, row in df_basvurular.iterrows():

        pozisyon = row["Pozisyon"]
        # Eğer başvurulan pozisyon şirket pozisyonları içinde yoksa devam et
        if pozisyon not in df_sirket_pozisyonlari["Pozisyon"].tolist():
            continue

        başvuran_tfidf_vector = tfidf_matrix[başvuran_aday_sayacı[index]]

        # Şirket pozisyonları içerisindeki pozisyonun indeksinin bulunması
        pozisyon_indeks = df_sirket_pozisyonlari.index[df_sirket_pozisyonlari["Pozisyon"] == pozisyon].tolist()[0]

        # Şirket pozisyonunun TF-IDF vektörünün alınması
        şirket_tfidf_vector = tfidf_matrix[pozisyon_indeks]
        # Cosine benzerliğinin hesaplanması
        benzerlik_skoru = cosine_similarity(başvuran_tfidf_vector, şirket_tfidf_vector)[0][0]
        x=benzerlik_skoru*100
        x=int(x)
        sonuç_df = pd.concat(
            [sonuç_df, pd.DataFrame({"Başvuran Kişi": [row["Ad"] + " " + row["Soyad"]],"Email":[row["Eposta"]],"Başvurulan Pozisyon": [pozisyon],
                                     "Pozisyonun Uygunluk Oranı %": [x]})], ignore_index=True)
        with open('model.pkl', 'wb') as f:
            pickle.dump(sonuç_df, f)

        with open('similarity.pkl', 'wb') as f:
            pickle.dump(benzerlik_skoru, f)

    sonuç_df["Pozisyon Uygunluğu"] = ["Elemeden Geçildi ✓" if x >= uygunluk_skoru else "Elemeden Geçemedi X" for x in
                                      sonuç_df["Pozisyonun Uygunluk Oranı %"]]

    sonuç_df.drop_duplicates(subset=["Başvuran Kişi", "Başvurulan Pozisyon"], keep='first', inplace=True)

    sonuç_df = sonuç_df.sort_values(by="Pozisyonun Uygunluk Oranı %", ascending=False)

    return sonuç_df.to_html(index=False, escape=False)

@app.route('/download_excel', methods=['POST'])
def download_excel():
    global sonuç_df

    if sonuç_df is None:
        return "Önce sonuçları değerlendirin."

    # Formdan gelen pozisyon filtresini al
    filter_position = request.form.get('filter_position', '')

    # DataFrame'i pozisyona göre filtrele
    if filter_position:
        filtered_df = sonuç_df[sonuç_df['Başvurulan Pozisyon'] == filter_position]
    else:
        filtered_df = sonuç_df

    # Geçici bir dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
        excel_file_path = temp.name

    # DataFrame'i Excel dosyasına yaz
    filtered_df.to_excel(excel_file_path, index=False)

    # Excel dosyasını indir
    return send_file(excel_file_path, as_attachment=True, download_name='başvuru_sonuçları.xlsx')



# Ön işleme fonksiyonları
def turkce_kucuk_harf(metin):
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    return metin

@app.route('/gorsellestirme')
def gorsellestirme():
    if 'user' in session:

        df_basvurular=adaylar()
        if df_basvurular is None:
            return "No data available", 404

        basvurular = df_basvurular.copy()
        # Veri ön işleme
        basvurular["Pozisyon"] = basvurular["Pozisyon"].apply(turkce_kucuk_harf)

        wordcloud_fig = wordcloud_grafik(basvurular)
        # Grafikleri oluşturun
        grafikler = {
            'pozisyon_dagilimi': pozisyon_dagilimi_grafik(basvurular),
            'Yabancı_Dil_grafik': Yabancı_Dil_grafik(basvurular),
            'is_deneyimi': is_deneyimi_grafik(basvurular),
            'pozisyon_egitim': pozisyon_egitim(basvurular),
            'pozisyon_teknoloji': pozisyon_teknoloji(basvurular),
            'pozisyon_egitim_is': pozisyon_egitim_is(basvurular),
            'wordcloud': wordcloud_fig,
            'eğitim_seviyesi': egitim_seviyesi(basvurular),
            'ikametgah_dagilimi': ikametgah_dagilimi(basvurular),
            'yabanci_dil_seviyesi': yabanci_dil_seviyesi_grafik(basvurular)
        }

        # Grafikleri JSON formatına dönüştürün
        grafikler_json = {k: json.dumps(v, cls=plotly.utils.PlotlyJSONEncoder) for k, v in grafikler.items()}
        return render_template('Görselleştirme.html', grafikler=grafikler_json)

    return redirect(url_for('login'))


def pozisyon_dagilimi_grafik(bs):
    fig = px.bar(bs, x='Pozisyon', title='Pozisyon Dağılımı', color_discrete_sequence=['orange'])
    return fig



def Yabancı_Dil_grafik(bs):
    fig = px.pie(bs, names='Yabancı Dil', title='Başvuranların Yabancı Dil Dağılımı')
    return fig


def is_deneyimi_grafik(bs):
    avg_experience = bs.groupby('Pozisyon')['İş Deneyimi (Yıl)'].mean().reset_index()
    fig = px.bar(avg_experience, x='Pozisyon', y='İş Deneyimi (Yıl)', title='Pozisyonlara Göre Ortalama İş Deneyimi', color_discrete_sequence=['green'])
    return fig


def pozisyon_egitim(bs):
    grouped = bs.groupby(['Pozisyon', 'Eğitim Seviyesi']).size().reset_index(name='counts')
    fig = px.bar(grouped, x='Pozisyon', y='counts', color='Eğitim Seviyesi', title='Pozisyonlara Göre Başvuranların Eğitim Seviyesi Dağılımı')

    return fig

def egitim_seviyesi(bs):
    fig = px.bar(bs, x='Eğitim Seviyesi', title='Başvuranların Eğitim Seviyesi Dağılımı', color_discrete_sequence=['red'])

    return fig

def pozisyon_teknoloji(bs):
    position_technologies = bs[['Pozisyon', 'Teknolojiler']].copy()
    position_technologies['Teknolojiler'] = position_technologies['Teknolojiler'].str.split(',')
    position_technologies = position_technologies.explode('Teknolojiler')
    grouped = position_technologies.groupby(['Pozisyon', 'Teknolojiler']).size().reset_index(name='counts')
    fig = px.bar(grouped, x='Pozisyon', y='counts', color='Teknolojiler', title='Pozisyon ve Teknoloji İlişkisi')
    return fig


def pozisyon_egitim_is(bs):
    grouped = bs.groupby(['Pozisyon', 'Eğitim Seviyesi'])['İş Deneyimi (Yıl)'].mean().reset_index()
    fig = px.bar(grouped, x='Pozisyon', y='İş Deneyimi (Yıl)', color='Eğitim Seviyesi', title='Pozisyonlara ve Eğitim Seviyelerine Göre İş Deneyimi')
    return fig

def ikametgah_dagilimi(bs):
    fig = px.bar(bs, x='İkametgah', title='Başvuranların İkametgah Dağılımı', color_discrete_sequence=['purple'])
    return fig

def yabanci_dil_seviyesi_grafik(bs):
    grouped = bs.groupby(['Yabancı Dil', 'Yabancı Dil Seviyesi']).size().reset_index(name='counts')
    fig = px.bar(grouped, x='Yabancı Dil', y='counts', color='Yabancı Dil Seviyesi', title='Yabancı Dil ve Seviyelerine Göre Dağılım', barmode='stack')
    return fig


def wordcloud_grafik(bs):
    if bs.empty:
        return None  # Eğer bs boşsa hiçbir işlem yapmadan çık

    text = ' '.join(bs['Teknolojiler'].dropna().str.split(',').sum())
    wc = WordCloud(background_color='white', width=800, height=400,max_words=1000,contour_width=3,
               contour_color="firebrick").generate(text)

    # Matplotlib figürünü Plotly figürüne dönüştür
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')

    # Matplotlib figürünü geçici bir dosyaya kaydet ve oku
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Plotly figürü oluştur
    plotly_fig = {
        'data': [{
            'type': 'image',
            'source': f'data:image/png;base64,{img_str}',
            'x': 0,
            'y': 0,
            'sizex': 1,
            'sizey': 1,
            'xanchor': 'left',
            'yanchor': 'bottom'
        }],
        'layout': {
            'title': 'Teknolojiler Word Cloud',
            'xaxis': {'visible': False},
            'yaxis': {'visible': False}
        }
    }
    return plotly_fig

@app.route('/')
def index():
    df_sirket_pozisyonlari = sirket()
    cities_file = os.path.join(app.root_path, 'static', 'cities.json')

    with open(cities_file, 'r', encoding='utf-8') as json_file:
        cities = json.load(json_file)['cities']

    return render_template('Anasayfa.html', cities=cities)


@app.route('/aday_degerlendir')
def aday_degerlendir():
    if 'user' in session:
        return render_template('aday_değerlendirme.html')

    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        df_basvurular=adaylar()
        df_sirket_pozisyonlari=sirket()
        # Açılan ilan sayısı
        ilan_sayisi = len(df_sirket_pozisyonlari['Pozisyon'].unique())

        # Başvuran aday sayısı
        aday_sayisi = len(df_basvurular)

        # Favori ilan
        favori_ilan = df_basvurular['Pozisyon'].value_counts().idxmax()

        return render_template('dashboard.html', ilan_sayisi=ilan_sayisi, aday_sayisi=aday_sayisi, favori_ilan=favori_ilan)

    return redirect(url_for('login'))

@app.route('/acilan_ilanlar')
def acilan_ilanlar():
    if 'user' in session:
        return render_template('acilan_ilanlar.html')

    return redirect(url_for('login'))

@app.route('/get_company_positions', methods=['POST'])
def get_company_positions():
    df_sirket_pozisyonlari=sirket()
    return df_sirket_pozisyonlari[
        ["Pozisyon", "Teknolojiler", "Eğitim Seviyesi","Çalışma Şekli","Yabancı Dil", "Yabancı Dil Seviyesi",
         "Aday Profili"]].to_html(index=False, escape=False,classes='table table-striped table-bordered')

@app.route('/get_positions',methods=["POST"])
def get_positions():
    df_sirket_pozisyonlari=sirket()
    return df_sirket_pozisyonlari["Pozisyon"].to_json(orient='values')

@app.route('/skills', methods=['GET'])
def get_skills():
    skills_file = os.path.join(app.root_path, 'static', 'skills.json')
    try:
        with open(skills_file, 'r', encoding='utf-8') as file:
            skills_data = json.load(file)
        return jsonify(skills_data)
    except Exception as e:
        return str(e), 500  # Hata durumunda 500 (Internal Server Error) döndür

@app.route('/remove_position')
def ilan_kaldır():
    if 'user' in session:
        return render_template('remove_position.html')
    return redirect(url_for('login'))

@app.route('/remove_position', methods=['POST'])
def remove_position():

    # Form verilerini al
    position_name = request.form.get('position_name')

    # İlgili pozisyonu bulun
    position = SirketPozisyon.query.filter_by(pozisyon=position_name).first()

    # Pozisyonu kaldırın
    if position:
        db.session.delete(position)
        db.session.commit()
        message = "{} Pozisyonu Başarıyla Kaldırıldı.".format(position_name)
    else:
        message = "Bu Pozisyon Veri Setinde Bulunmamaktadır."

    return render_template('remove_position.html', message=message)



@app.route('/ilan_ekle')
def ilan_ekle():
    if 'user' in session:
        return render_template('ilan_ekle.html')
    return redirect(url_for('login'))

@app.route('/değerlendirme_hakkında')
def değerlendirme_hakkında():
    return render_template('değerlendirme_hakkında.html')

class SirketPozisyon(db.Model):
    __tablename__ = 'sirket_pozisyonlari'  # Veritabanındaki tablo adı

    id = db.Column(db.Integer, primary_key=True)
    pozisyon = db.Column(db.String(100))
    teknolojiler = db.Column(db.String(1000))
    calisma_sekli = db.Column(db.String(100))
    yabanci_dil = db.Column(db.String(100))
    yabanci_dil_seviyesi = db.Column(db.String(100))
    aday_profil = db.Column(db.String(1000))
    egitim_seviyesi = db.Column(db.String(100))

@app.route('/ilan_ekle', methods=['POST'])
def ilan_ekle_post():


    if request.method == 'POST':
        formData = request.form

        # Gelen form verilerini al
        pozisyon = formData['pozisyon']
        teknolojiler = formData['selectedSkills']
        calisma_sekli=formData['calisma_sekli']
        yabanci_dil = formData['yabanci_dil']
        yabanci_dil_seviyesi = formData['yabanci_dil_seviyesi']
        aday_profil = formData['aday_profil']
        egitim_seviyesi = formData['egitim_seviyesi']

        # Yeni ilanı oluştur
        yeni_ilan = SirketPozisyon(
            pozisyon=pozisyon,
            teknolojiler=teknolojiler,
            calisma_sekli=calisma_sekli,
            yabanci_dil=yabanci_dil,
            yabanci_dil_seviyesi=yabanci_dil_seviyesi,
            aday_profil=aday_profil,
            egitim_seviyesi=egitim_seviyesi
        )


        db.session.add(yeni_ilan)
        db.session.commit()

        if 'user' in session:
            return render_template('ilan_ekle.html', message='{} İlanı Başarıyla Eklendi.'.format(pozisyon))
        return redirect(url_for('login'))

# Başvuru modeli
class Basvuru(db.Model):
    __tablename__ = 'adayOzgecmis'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    number = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(50),nullable=False)
    resume = db.Column(db.LargeBinary, nullable=False)

class Adaylar(db.Model):
    __tablename__ = 'basvurular'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    surname = db.Column(db.String(100))
    dob = db.Column(db.Date)
    education = db.Column(db.String(100))
    position = db.Column(db.String(100))
    experience = db.Column(db.Integer)
    selectedSkills = db.Column(db.String(100))
    languages = db.Column(db.String(100))
    languageslevel = db.Column(db.String(100))
    intro = db.Column(db.String(500))
    location = db.Column(db.String(100))
    number = db.Column(db.String(20))
    email = db.Column(db.String(100))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/basvur', methods=['POST'])
def basvur():
    df_basvurular=adaylar()

    if request.method == 'POST':
        formData = request.form

        # Başvuru verilerini işle
        if ((df_basvurular['Ad'] == formData['name']) & (df_basvurular['Soyad'] == formData['surname']) & (
                df_basvurular['Doğum Tarihi'] == formData['dob']) & \
            (df_basvurular['Eğitim Seviyesi'] == formData['education']) & (
                    df_basvurular['Pozisyon'] == formData['position']) & \
            (df_basvurular['İş Deneyimi (Yıl)'] == formData['experience']) & (
                    df_basvurular['Teknolojiler'] == formData['selectedSkills']) & \
            (df_basvurular['Yabancı Dil'] == formData['languages']) & (

            df_basvurular['İkametgah'] == formData['location']) & (

            df_basvurular['Telefon Numarası'] == formData['number']) & (

            df_basvurular['Eposta'] == formData['email']) & (
                    df_basvurular['Aday Profili'] == formData['intro']) & (
                    df_basvurular['Yabancı Dil Seviyesi'] == formData['languageslevel'])).any():
            return render_template('Anasayfa.html', message='Bu başvuru zaten yapılmıştır.')

        new_data = pd.DataFrame({'Ad': [formData['name']],
                                 'Soyad': [formData['surname']],
                                 'Doğum Tarihi': [formData['dob']],
                                 'Eğitim Seviyesi': [formData['education']],
                                 'Pozisyon': [formData['position']],
                                 'İş Deneyimi (Yıl)': [formData['experience']],
                                 'Teknolojiler': [formData['selectedSkills']],
                                 'Yabancı Dil': [formData['languages']],
                                 'Yabancı Dil Seviyesi': [formData['languageslevel']],
                                 'Aday Profili': [formData['intro']],
                                 'İkametgah': [formData['location']],
                                 'Telefon Numarası': [formData['number']],
                                 'Eposta': [formData['email']],
                                 })
        cv = request.files['resume']
        cv_data = cv.read()  # Dosya içeriğini oku

        # Yeni başvuru oluştur
        new_document = Basvuru(
            name=formData['name'],
            surname=formData['surname'],
            number=formData['number'],
            email=formData['email'],
            position=formData['position'],
            resume=cv_data  # Dosya içeriğini kaydet
        )
        # Veritabanına kaydet
        db.session.add(new_document)
        db.session.commit()

        # Yeni başvuru oluştur
        new_Aday = Adaylar(
            name=formData['name'],
            surname=formData['surname'],
            dob=formData['dob'],
            education=formData['education'],
            position=formData['position'],
            experience=formData['experience'],
            selectedSkills=formData['selectedSkills'],
            languages=formData['languages'],
            languageslevel=formData['languageslevel'],
            intro=formData['intro'],
            location=formData['location'],
            number=formData['number'],
            email=formData['email']
        )

        # Veritabanına kaydet
        db.session.add(new_Aday)
        db.session.commit()


        return render_template('Anasayfa.html', message='{} {} Başvurunuz Alınmıştır, Teşekkür ederiz!'.format(formData['name'],formData['surname']))

@app.route('/basvurular', methods=['GET'])
def basvurular():
    if 'user' in session:
        basvuru_listesi = Basvuru.query.all()
        return render_template('aday_ozgecmis.html', basvurular=basvuru_listesi)
    return redirect(url_for('login'))

@app.route('/Aday_basvuruları',methods=['GET'])
def aday_basvuruları():
    if 'user' in session:
        return render_template('Aday_basvuruları.html')
    return redirect(url_for('login'))

@app.route('/get_company_users', methods=['POST'])
def get_company_users():
    df_basvurular=adaylar()
    return df_basvurular[
        ["Ad", "Soyad", "Doğum Tarihi", "Eğitim Seviyesi", "Pozisyon", "İş Deneyimi (Yıl)",
                         "Teknolojiler", "Yabancı Dil", "Yabancı Dil Seviyesi", "İkametgah",
                         "Telefon Numarası", "Eposta","Aday Profili"]].to_html(index=False, escape=False,classes='table table-striped table-bordered')

@app.route('/download_resume/<int:id>', methods=['GET'])
def download_resume(id):
    basvuru = Basvuru.query.get(id)
    if not basvuru or not basvuru.resume:
        return "Resume not found", 404

    resume_data = basvuru.resume

    # Binary veriyi BytesIO nesnesine dönüştür
    resume_stream = BytesIO(resume_data)

    # Dosyayı indirme olarak döndür
    return send_file(
        resume_stream,
        as_attachment=True,
        download_name=f"{basvuru.name}_{basvuru.surname}_resume.pdf",
        mimetype='application/pdf'
    )

@app.route('/view_resume/<int:id>', methods=['GET'])
def view_resume(id):
    basvuru = Basvuru.query.get(id)
    if not basvuru or not basvuru.resume:
        return "Resume not found", 404

    resume_data = basvuru.resume

    # Binary veriyi BytesIO nesnesine dönüştür
    resume_stream = BytesIO(resume_data)

    # Dosyayı PDF olarak döndür
    return send_file(
        resume_stream,
        mimetype='application/pdf'
    )

# CV'den metin çıkarma fonksiyonu
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    num_pages = len(reader.pages)
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Ön işleme adımlarını içeren fonksiyon
def preprocess_text(text):
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.remove('not')  # 'not' kelimesini stop word listesinden kaldırma
    processed_text = re.sub('[^a-zA-Z]', ' ', text)  # Sadece harf karakterlerini koru
    processed_text = processed_text.lower()  # Küçük harfe dönüştür
    processed_text = processed_text.split()  # Kelimelere ayır
    processed_text = [ps.stem(word) for word in processed_text if word not in stopwords_set]  # Stop kelimeleri kaldır ve köklerine ayır
    processed_text = ' '.join(processed_text)  # Tekrar birleştir
    return processed_text


# Tahmin fonksiyonu
def predict_job(extracted_text):
    # Modeli yükleme
    loaded_model = joblib.load('logistic_regression_model.pkl')

    # TF-IDF vektörlerini yükleme
    tfidf_v = joblib.load('tfidf_vectorizer.pkl')

    # Örnek metni ön işleme adımlarından geçirme
    processed_extracted_text = preprocess_text(extracted_text)

    # TF-IDF vektörleştirme
    extracted_text_vectorized = tfidf_v.transform([processed_extracted_text]).toarray()

    # Model üzerinde tahmin yapma
    prediction = loaded_model.predict(extracted_text_vectorized)

    return prediction

# Ana sayfa ve tahmin işlemi
@app.route('/is_önerisi', methods=['GET', 'POST'])
def is_onerisi():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('is_önerisi.html', prediction="Özgeçmiş Dosyası Yüklenmedi")
        file = request.files['file']
        if file.filename == '':
            return render_template('is_önerisi.html', prediction="Özgeçmiş Dosyası Seçilmedi")
        if file:
            cv_text = extract_text_from_pdf(file)
            prediction = predict_job(cv_text)
            prediction_text = 'Özgeçmişinize Uygun Olarak Önerilen Pozisyon: ' + prediction[0] if prediction else "Öneri Bulunamadı"
            return render_template('is_önerisi.html', prediction=prediction_text)
    return render_template('is_önerisi.html')

def get_city_coordinates(city_names):
    geolocator = Nominatim(user_agent="city_locator", timeout=10)
    coordinates = []
    for city in city_names:
        location = geolocator.geocode(city)
        if location:
            coordinates.append((city, location.latitude, location.longitude))
        else:
            print("Koordinatlar alınamadı:", city)
    return coordinates


@app.route('/başvurulan_lokasyonlar')
def lokasyonlar():
    df_basvurular=adaylar()
    if df_basvurular is None:
        return "No data available", 404

    if 'user' in session:
        ikametgah_values = df_basvurular['İkametgah']
        cities = [city.lower() for city in ikametgah_values.tolist()]
        city_counts = Counter(cities)
        city_coordinates = get_city_coordinates(cities)

        map_center = [39.9334, 32.8597]  # Ankara merkezi
        mymap = folium.Map(location=map_center, zoom_start=6)

        # Şehirler için marker'ları ekle
        for city, lat, lon in city_coordinates:
            candidates = df_basvurular[df_basvurular['İkametgah'].str.lower() == city]
            candidate_details = "<br>".join([
                f"Ad: {row['Ad']} {row['Soyad']}<br>Pozisyon: {row['Pozisyon']}<br>"
                for idx, row in candidates.iterrows()
            ])
            popup_content = f"""
                <hr style="margin: 5px;">
                <div style="text-align: center; font-weight: bold;">
                    {city.capitalize()}
                </div>
                <br>
                <div>
                    <strong>Başvuran aday sayısı:</strong> {city_counts[city]}
                </div>
                <br>
                <div>
                    {candidate_details}
                </div>
            """
            iframe = folium.IFrame(html=popup_content, width=300, height=180)
            popup = folium.Popup(iframe, max_width=300)
            folium.Marker(location=[lat, lon], popup=popup,
                          icon=folium.Icon(color='red', icon='info-sign'),tooltip="Başvuruyu Gör").add_to(mymap)

        # Şirketin lokasyonu için özel marker ve popup ekleyelim
        company_location = [41.01406875375204, 29.204855945968234]
        with open("static/i.png", "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        company_popup_content = f"""
            <hr style="margin: 5px;">
            <div style="text-align: center; font-weight: bold;">
                İstanbul Çekmeköy
            </div>
            <br>
            <img src="data:image/png;base64,{encoded_image}" alt="Logo" style="max-width: 200px;">
            <br>
            <br>
             <div style="text-align: center;">
                 Hasan Can Bilişim Teknolojileri A.Ş.
            </div>
        """
        company_iframe = folium.IFrame(html=company_popup_content, width=300, height=150)
        company_popup = folium.Popup(company_iframe, max_width=300)
        folium.Marker(location=company_location, popup=company_popup, icon=folium.Icon(color='blue'),tooltip="Şirket Konumu").add_to(mymap)
        folium.TileLayer('openstreetmap').add_to(mymap)

        map_data = mymap.get_root().render()
        return render_template("başvurulan_lokasyonlar.html", map_data=map_data)
    return redirect(url_for('login'))



if __name__ == '__main__':
    app.run(debug=True)

