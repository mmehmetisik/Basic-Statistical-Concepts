
######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme)
############################

# "Without a grounding in Statictics, a Data Scientist is a Data Lab Assistant."

# Sampling (Örnekleme) = Bir ana kitle içerisinden bu ana kitlenin özelliklerini iyi taşıdığı(temsil ettiği) var sayılan
# bir alt kümedir. Ana kitlenin(kütlenin) temsilcisidir.

# "The Future of AI Will Be About Less Data, Not More"

populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################
# Ana kitle(kütle) parametresinin tahmini değerini (istatistik) kapsayabilecek iki sayıdan oluşan bir aralık
# bulunmasıdır.

# Örneğin:

# Bir web sitesinde geçirilen ortalama sürenin güven aralığı nedir ?

# ortalama: 180 saniye
# Standart sapma: 40 saniye

# Güven aralığı: %95 güven ile 172 - 188 arasındadır.

# Adım 1: n, ortalama ve standart sapmayı bul.
n = 100
ortalama = 180
standart_sapma = 40

# Adım 2: Güven Aralığına Karar Ver: %95 mi %99 mu ?

# Z tablo değerini hesapla (1,96 - 2,57)
# Genel de  %95 alınırı.
# Z tablo değeri de 1,96 olarak seçildi.

# Adım 3: Yukarıdaki değerleri kullanarak güven aralığını hesapla

# Sonuç:
# 180 +- 7,84 yani 172 ile 188 arasındadır.

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean() # total_bill değişkeninin ortalaması istatistiki olarak %95 güven ile
# x ile y değerleri arasındadır. %5 hata payı vardır.

sms.DescrStatsW(df["tip"]).tconfint_mean() # tip değişkeninin ortalaması istatistiki olarak %95 güven ile
# x ile y değerleri arasındadır. %5 hata payı vardır.

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean() # age değişkeninin ortalaması istatistiki olarak %95 güven ile
# x ile y değerleri arasındadır. %5 hata payı vardır.

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean() # fare değişkeninin ortalaması istatistiki olarak %95 güven ile
# x ile y değerleri arasındadır. %5 hata payı vardır.


######################################################
# Correlation (Korelasyon)
######################################################
#  Değişkenler arasındaki ilişki, bu ilişkinin yönü ve şiddeti ile ilgili bilgiler sağlayan istatistiksel bir yöntemdir.
# 1 ile -1 arasında yer alır. 0 korelasyon yok demektir.
# Pozitif 1 tarafına gidildikçe Mükemmel Pozitif Korelasyon, -1 tarafına doğru gidildikçe Mükemmel Negatif Korelasyon
# ortaya çıkar.

# Pozitif Korelasyon: Bir değişkenin değeri artarken diğer değişkenin değerinin de artacağı anlamına gelir.
# Negatif Korelasyon: Bir değişkenin değeri artarken diğer değişkenin değerinin azalacağı anlamına gelir.

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

# Toplam hesap ile ödenen bahşiş arasında bir kolerasyon var mı ?
df.plot.scatter("tip", "total_bill")
plt.show(block=True)

df["tip"].corr(df["total_bill"])

# Toplam hesap ile ödenen bahşiş arasında pozitif yönlü bir korelasyon vardır.
