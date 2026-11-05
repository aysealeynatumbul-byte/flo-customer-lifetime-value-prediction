##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# ADIM 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.


import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option("display.max_columns", None)

df_ = pd.read_csv(r"C:\Users\HP\PycharmProjects\PythonProject\CLTV_FLO\flo_data_20k.csv")
df_.head(10)

df = df_.copy()
df.head()
df.describe().T



# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
 # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)   # en düşük %1 sınırı / “Verinin %1’lik kısmı hangi değerin altında kalıyor?”
    quartile3 = dataframe[variable].quantile(0.99)   # en yüksek %1 sınırı / “Verinin %99’u hangi değerin altında kalıyor?”
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)
# 0.01 ve 0.99 kullanınca, gerçekten yüksek ama mantıklı müşterileri korumuş oluyoruz.
# Sadece aşırı uçtaki, modeli bozabilecek çok büyük değerleri baskılıyoruz.

# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlıyoruz.


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
#CLTV hesaplamasında aşırı sipariş ve harcama değerlerinin modeli yanıltmasını engellemek için baskılama (winsorization) yaparız
#Böylece  müşteri veri setinde kalır ve müşteri kaybetmemiş oluruz.


replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")


 # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["order_num_total_ever"].head()

df["customer_value_total_ever"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["customer_value_total_ever"].head()


 # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

import datetime as dt

df.info()
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df.dtypes



# ADIM 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()   #'2021-05-30'
today_date = dt.datetime(2021, 6 ,1)

 # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.



cltv_df = df.groupby("master_id").agg({
    "first_order_date": "min",
    "last_order_date": "max",
    "order_num_total_ever": "sum",
    "customer_value_total_ever": "sum"
})

cltv_df = cltv_df.rename(columns={
    "first_order_date": "first_order_date",
    "last_order_date": "last_order_date",
    "order_num_total_ever": "frequency",
    "customer_value_total_ever": "monetary"
})

cltv_df.head()

# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

# İlk alışveriş ile son alışveriş arasındaki süre
cltv_df["recency_cltv_weekly"] = (
    cltv_df["last_order_date"] - cltv_df["first_order_date"]
).dt.days / 7


# İlk alışveriş ile analiz tarihi arasındaki süre
cltv_df["T_weekly"] = (
    today_date - cltv_df["first_order_date"]
).dt.days / 7


# Toplam sipariş sayısının integer olduğundan emin olma
cltv_df["frequency"] = cltv_df["frequency"].astype(int)
# Toplam alışveriş sayısı 1'den büyük müşterileri seçeriz
cltv_df = cltv_df[cltv_df["frequency"] > 1]


# Sipariş başına ortalama harcama
cltv_df["monetary_cltv_avg"] = (
    cltv_df["monetary"]
    / cltv_df["frequency"]
)


# ADIM 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması


# 1. BG/NBD modelini fit ediniz.
##BG/NBDmodeli ile  müşterinin geçmiş satın alma sıklığına
# ve son alışverişinin ne kadar güncel olduğuna bakarak gelecekteki
# tekrar satın alma sayısını tahmin edilir



# BG/NBD modelinin kurulması
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Modeli eğitiyoruz
bgf.fit(
    cltv_df["frequency"],
    cltv_df["recency_cltv_weekly"],
    cltv_df["T_weekly"]
)

# 3 aylık beklenen satın alma sayısı
cltv_df["exp_sales_3_month"] = (
    bgf.conditional_expected_number_of_purchases_up_to_time(
        t=4 * 3,
        frequency=cltv_df["frequency"],
        recency=cltv_df["recency_cltv_weekly"],
        T=cltv_df["T_weekly"]
    )
)

# 6 aylık beklenen satın alma sayısı
cltv_df["exp_sales_6_month"] = (
    bgf.conditional_expected_number_of_purchases_up_to_time(
        t=4 * 6,
        frequency=cltv_df["frequency"],
        recency=cltv_df["recency_cltv_weekly"],
        T=cltv_df["T_weekly"]
    )
)



# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

#GammaGamma ile "Müşteri her alışverişte ortalama ne kadar harcar?" sorusu yanıtlanır.
#Gamma-Gamma için monetary değeri, müşterinin toplam harcaması değil, işlem başına ortalama
#harcaması olmalıdır.
#Gamma-Gamma modeli, bu müşterinin gelecekteki işlemlerinde ortalama ne kadar harcayacağını tahmin eder.


ggf = GammaGammaFitter(penalizer_coef = 0.001)
ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"],)

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                   cltv_df["monetary_cltv_avg"])

# exp_average_value adlı sütun oluşturlurken işlem başına ortalama
# harcaması yani monetary_cltv_avg kullanılır çünkü GammaGamma modeli ortalama harcamayı kullanır.


# Sipariş başına ortalama harcama
cltv_df["monetary_cltv_avg"] = (
    cltv_df["monetary"]
    / cltv_df["frequency"]
)


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#BG/NBD ve GammaGamma modeli ikisi birlikte kullanılarak
#Müşterinin gelecek 6 ayda şirkete bırakacağı toplam değer, yani CLTV hesaplanır.


cltv = ggf.customer_lifetime_value(
    bgf,
    cltv_df["frequency"],
    cltv_df["recency_cltv_weekly"],
    cltv_df["T_weekly"],
    cltv_df["monetary_cltv_avg"],
    time=6,
    freq="W",
    discount_rate=0.01
)


# CLTV sonuçlarını tabloya dönüştürme
cltv = cltv.rename("clv").reset_index()

# Müşteri bilgileri ile CLTV değerlerini birleştirme
cltv_final = cltv_df.reset_index().merge(
    cltv,
    on="master_id",
    how="left"
)

# En yüksek CLTV değerine sahip 10 müşteri
print(cltv_final.sort_values(
    by="clv", ascending=False
).head(10))


# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.


cltv_final.sort_values(by="clv", ascending=False).head(20)



# ADIM 4: CLTV'ye Göre Segmentlerin Oluşturulması


# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=[ "D" ,"C","B","A"])












