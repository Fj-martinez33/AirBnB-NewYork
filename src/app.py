#Utils.py --->
# from utils import db_connect
#engine = db_connect()

# Librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np
import json
from pickle import dump

#Recopilar datos
data = pd.read_csv("../data/raw/AB_NYC_2019.csv")

#Obtener las dimensiones
print(data.shape)

#Obtener información sobre tipo de datos y valores nulos.
print(data.info())

#Funcion para eliminar duplicados
#Columna identificadora del Dataset.
def EraseDuplicates(dataset, id = "id"):
    older_shape = dataset.shape
    if (dataset.drop(id, axis = 1).duplicated().sum()):
        print ("Erase duplicates...")
        dataset.drop(id, axis = 1, inplace = True)
        dataset.drop_duplicates()
        print(f"Total number of duplicates {dataset.duplicated().sum()}")
    else:
        print ("No coincidences.")
        dataset.drop(id, axis=1, inplace = True)
        pass
    
    print (f"The older dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    
    return dataset

EraseDuplicates(data)

print(data)

#Funcion para eliminar datos irrelevantes.
irrelevant_lst = ["name","host_id","host_name","neighbourhood","latitude","longitude","last_review","reviews_per_month"]

def EraseIrrelevants(dataset, lst):
    older_shape = data.shape
    print("Erase irrelevant´s dates...")
    dataset.drop(lst, axis = 1, inplace = True)
    print (f"The old dimension of dataset is {older_shape}, and the new dimension is {dataset.shape}.")
    return dataset

EraseIrrelevants(data, irrelevant_lst)

print(data)

# Analisis sobre variables categoricas
def CategoricGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(1, 3, figsize=(15,5))

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0], data = dataset, x = "neighbourhood_group")
    sns.histplot( ax = axis[1], data = dataset, x = "room_type").set(ylabel = None)
    sns.histplot( ax = axis[2], data = dataset, x = "availability_365").set(ylabel = None)

    #Mostramos el grafico.
    plt.tight_layout()
    plt.show()

CategoricGraf(data)

# Analisis sobre variables numericas
def NumericalGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(4, 2, figsize=(10,8), gridspec_kw={"height_ratios" : [6,1,6,1]})

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0,0], data = dataset, x = "price", kde=True).set(xlabel=None)
    sns.boxplot( ax = axis[1,0], data = dataset, x = "price")
    sns.histplot( ax = axis[0,1], data = dataset, x = "minimum_nights", kde=True).set(ylabel=None, xlabel=None)
    sns.boxplot( ax = axis[1,1], data = dataset, x = "minimum_nights")
    sns.histplot( ax = axis[2,0], data = dataset, x = "number_of_reviews").set(xlabel=None)
    sns.boxplot(ax = axis[3,0], data = dataset, x = "number_of_reviews")
    sns.histplot( ax = axis[2,1], data = dataset, x = "calculated_host_listings_count", kde=True).set(ylabel=None, xlabel=None)
    sns.boxplot( ax = axis[3,1], data = dataset, x = "calculated_host_listings_count")
    
    plt.tight_layout()
    plt.show()

NumericalGraf(data)

def FiltNumericalGraf(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(4, 2, figsize=(10,8), gridspec_kw={"height_ratios" : [6,1,6,1]})

    #Creamos las graficas necesarias
    sns.histplot( ax = axis[0,0], data = dataset[dataset["price"] < 500], x = "price", kde=True).set(xlabel=None)
    sns.boxplot( ax = axis[1,0], data = dataset[dataset["price"] < 500], x = "price")
    sns.histplot( ax = axis[0,1], data = dataset[dataset["minimum_nights"] < 20], x = "minimum_nights", kde=True).set(ylabel=None, xlabel=None)
    sns.boxplot( ax = axis[1,1], data = dataset[dataset["minimum_nights"] < 20], x = "minimum_nights")
    sns.histplot( ax = axis[2,0], data = dataset[dataset["number_of_reviews"] < 20], x = "number_of_reviews", kde=True).set(xlabel=None, ylabel=None)
    sns.boxplot(ax = axis[3,0], data = dataset[dataset["number_of_reviews"] < 20], x = "number_of_reviews")
    sns.histplot( ax = axis[2,1], data = dataset[dataset["calculated_host_listings_count"] < 10], x = "calculated_host_listings_count", kde=True).set(ylabel=None, xlabel=None)
    sns.boxplot( ax = axis[3,1], data = dataset[dataset["calculated_host_listings_count"] < 10], x = "calculated_host_listings_count")
    
    plt.tight_layout()
    plt.show()

FiltNumericalGraf(data)

#Analisis numerico/numerico
def NumNumAnalysi(dataset, x, y_list):
    #Creamos la figura
    fig, axis = plt.subplots(2, 3, figsize=(15,8))

    #Creamos la grafica
    sns.regplot( ax = axis[0,0], data = dataset, x = y_list[0], y = x)
    sns.heatmap( data[[x,y_list[0]]].corr(), annot=True, fmt=".2f", ax = axis[1,0], cbar=False)
    sns.regplot( ax = axis[0,1], data = dataset, x = y_list[1], y = x).set(ylabel=None)
    sns.heatmap( data[[x,y_list[1]]].corr(), annot=True, fmt=".2f", ax = axis[1,1], cbar=False)
    sns.regplot( ax = axis[0,2], data = dataset, x = y_list[2], y = x).set(ylabel=None)
    sns.heatmap( data[[x, y_list[2]]].corr(), annot=True, fmt=".2f", ax = axis[1,2])

    plt.tight_layout()
    plt.show()

    #Creamos una segunda figura
    fig, axis = plt.subplots(2,2, figsize=(10,8))

    #Creamos la segunda grafica
    sns.regplot( ax = axis[0,0], data = dataset, x = y_list[0], y = y_list[1])
    sns.heatmap( data[[y_list[1], y_list[0]]].corr(), annot=True, fmt=".2f", ax = axis[1,0])
    sns.regplot( ax = axis[0,1], data = dataset, x = y_list[0], y = y_list[2])
    sns.heatmap( data[[y_list[2], y_list[0]]].corr(), annot=True, fmt=".2f", ax = axis[1,1])

    plt.tight_layout()
    plt.show()

NumNumAnalysi(data, "price", ["minimum_nights", "number_of_reviews","calculated_host_listings_count"])

def CatCatAnalysi(dataset):
    #Creamos la figura
    fig, axis = plt.subplots(1, 2, figsize=(10,5))

    #Creamos las graficas.
    sns.countplot(ax = axis[0], data = dataset, x = "neighbourhood_group", hue = "room_type")
    sns.countplot(ax = axis[1], data = dataset, x = "room_type", hue = "neighbourhood_group")

    plt.tight_layout()
    plt.show()

CatCatAnalysi(data)

#Convertimos las columnas categoricas al Dtype category para optimizar los datos.

def ObjectToCategory(dataset, col):
    dataset[col] = dataset[col].astype("category")
    return dataset


ObjectToCategory(data, "neighbourhood_group")
ObjectToCategory(data, "room_type")

data.info()

#Creamos el codificador

def Encoder(dataset, col):
    encoder = OneHotEncoder()
    encoding = encoder.fit_transform(dataset[[col]])
    dummie_col = pd.DataFrame(encoding.toarray(), columns=encoder.categories_)
    dataset = pd.concat([dataset, dummie_col], axis = 1)
    dataset.drop(col, axis=1, inplace=True)
    return dataset

data = (Encoder(data, "room_type"))
data = (Encoder(data, "neighbourhood_group"))

#Renombramos las columnas
data.rename(columns = {('Entire home/apt',) : "entire_home", ('Private room',) : "private_room", ('Shared room',) : "shared_room", ('Bronx',) : "bronx", ('Brooklyn',) : "brooklyn", ('Manhattan',) : "manhattan", ('Queens',) : "queens", ('Staten Island',) : "staten_island"}, inplace=True)

#Tabla de correlaciones
fig, axis = plt.subplots(figsize=(10,7))

sns.heatmap(data[["price","entire_home", "private_room", "shared_room", "bronx", "brooklyn", "manhattan", "queens", "staten_island", "minimum_nights","number_of_reviews","availability_365"]].corr(), annot=True, fmt=".2f")

plt.tight_layout()
plt.show()

#Corroboración de la tabla
fig, axis = plt.subplots(1,3,figsize=(15,5))

sns.regplot(ax = axis[0], data = data, x = "price", y = "entire_home")
sns.regplot(ax = axis[1], data = data, x = "price", y = "private_room")
sns.regplot(ax = axis[2], data = data, x = "price", y = "shared_room")

plt.tight_layout()
plt.show()

# Comprobamos las metricas de la tabla.
data.describe()

#Grafica de outliers

fig, axis = plt.subplots(2, 2, figsize=(10,5))

sns.boxplot( ax = axis [0,0], data = data, y = "price")
sns.boxplot( ax = axis [0,1], data = data, y = "minimum_nights")
sns.boxplot( ax = axis [1,0], data = data, y = "number_of_reviews")
sns.boxplot( ax = axis [1,1], data = data, y = "availability_365")

plt.tight_layout()
plt.show()

#Hacemos dos copias del dataset, una para el dataset con outliers y otra sin.
data_with_outliers = data.copy()
data_without_outliers = data.copy()

#Creamos una funcion para transformar los outliers.
def TransOutliers(dataset, col_outliers):
    stats = dataset[col_outliers].describe()
    
    #Establecemos los límites.
    # Los valores óptimos para sumarle al Q3 suelen ser 1.5*IQR, 1.75*IQR y 2*IQR.
    iqr = stats["75%"] - stats["25%"]
    upper_limit = float(stats["75%"] + (2 * iqr))
    lower_limit = float(stats["25%"] - (2 * iqr))
    
    if (lower_limit < 0):
        lower_limit = 0

    #Ajustamos el outlier por encima.
    dataset[col_outliers] = dataset[col_outliers].apply(lambda x : upper_limit if (x > upper_limit) else x)

    #Ajustamos el outlier por debajo.
    dataset[col_outliers] = dataset[col_outliers].apply(lambda x : lower_limit if (x < lower_limit) else x)

    #Guardamos los límites en un json.

    with open (f"../data/interim/outerliers_{col_outliers}.json", "w") as j:
        json.dump({"upper_limit" : upper_limit, "lower_limit" : lower_limit}, j)

    return dataset

TransOutliers (data_without_outliers, "minimum_nights")
TransOutliers (data_without_outliers, "number_of_reviews")
TransOutliers (data_without_outliers, "availability_365")

print(data_without_outliers)

#Comprobamos si existen valores faltantes.
data_with_outliers.isna().sum().sort_values()
data_without_outliers.isna().sum().sort_values()

# Primero dividimos los dataframes entre test y train

features = ["entire_home","private_room","shared_room","bronx","brooklyn","manhattan","queens","staten_island","minimum_nights","number_of_reviews","availability_365"]
target_feature = ["price"]

def SplitData (dataset, num_features, target):
    x = dataset.drop(target, axis = 1)[num_features]
    y = dataset[target].squeeze()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)

    return x_train, x_test, y_train, y_test

x_train_with_outliers, x_test_with_outliers, y_train, y_test = SplitData(data_with_outliers, features, target_feature)
x_train_without_outliers, x_test_without_outliers,_, _ = SplitData(data_without_outliers, features, target_feature)

y_train.to_excel("../data/processed/y_train.xlsx")
y_test.to_excel("../data/processed/y_test.xlsx")

#Tenemos que escalar los dataset con Normalizacion y con Escala mM (min-Max)

#Normalizacion
def StandardScaleData(dataset, num_features):
    scaler = StandardScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = num_features)
    
    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/standar_scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/standar_scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_standarscale = StandardScaleData(x_train_with_outliers, features)
x_train_without_outliers_standarscale = StandardScaleData(x_train_without_outliers,features)
x_test_with_outliers_standscale = StandardScaleData(x_test_with_outliers, features)
x_test_without_outliers_standscale = StandardScaleData(x_test_without_outliers, features)

#Escala mM
def MinMaxScaleData(dataset, num_features):
    scaler = MinMaxScaler()
    scaler.fit(dataset)

    x_scaler = scaler.transform(dataset)
    x_scaler = pd.DataFrame(dataset, index = dataset.index, columns = num_features)

    if(dataset is x_train_with_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_with_outliers.sav", "wb"))

    elif(dataset is x_train_without_outliers):
        dump(scaler, open("../data/interim/min-Max_Scale_without_outliers.sav", "wb"))

    return x_scaler

x_train_with_outliers_mMScaler = MinMaxScaleData(x_train_with_outliers, features)
x_train_without_outliers_mMScaler = MinMaxScaleData(x_train_without_outliers,features)
x_test_with_outliers_mMScaler = MinMaxScaleData(x_test_with_outliers, features)
x_test_without_outliers_mMScaler = MinMaxScaleData(x_test_without_outliers, features)

#Seleccion de caracteristicas
k = 11
def SelectFeatures(dataset, y, filename, k = k):
    sel_model = SelectKBest(f_classif, k=k)
    sel_model.fit(dataset, y)
    col_name = sel_model.get_support()
    x_sel = pd.DataFrame(sel_model.transform(dataset), columns = dataset.columns.values[col_name])
    dump(sel_model, open(f"../data/interim/{filename}.sav", "wb"))
    return x_sel

#Dataset sin normalizacion
x_train_sel_with_outliers = SelectFeatures(x_train_with_outliers, y_train, "x_train_with_outliers")
x_test_sel_with_outliers = SelectFeatures(x_test_with_outliers, y_test, "x_test_with_outliers")
x_train_sel_without_outliers = SelectFeatures(x_train_without_outliers, y_train, "x_train_without_outliers")
x_test_sel_without_outliers = SelectFeatures(x_test_without_outliers, y_test, "x_test_without_outliers")

#Dataset Normalizado
x_train_sel_with_outliers_standarscale = SelectFeatures(x_train_with_outliers_standarscale, y_train, "x_train_with_outliers_standarscale")
x_test_sel_with_outliers_standarscale = SelectFeatures(x_test_with_outliers_standscale, y_test, "x_test_with_outliers_standscale")
x_train_sel_without_outliers_standarscale = SelectFeatures(x_train_without_outliers_standarscale, y_train, "x_train_sel_without_outliers_standarscale")
x_test_sel_without_outliers_standarscale = SelectFeatures(x_test_without_outliers_standscale, y_test, "x_test_without_outliers_standscale")

#Dataset Escalado min-Max
x_train_sel_with_outliers_mMScale = SelectFeatures(x_train_with_outliers_mMScaler, y_train, "x_test_with_outliers_mMScaler")
x_test_sel_with_outliers_mMScale = SelectFeatures(x_test_with_outliers_mMScaler, y_test, "x_test_with_outliers_mMScaler")
x_train_sel_without_outliers_mMScale = SelectFeatures(x_train_without_outliers_mMScaler, y_train, "x_train_without_outliers_mMScaler")
x_test_sel_without_outliers_mMScale = SelectFeatures(x_test_with_outliers_mMScaler, y_test, "x_test_with_outliers_mMScaler")

#Para acabara añadimos el target a los datasets.
target = "price"
def AgreeTarget(dataset, y, target = target):
    dataset[target] = y
    return dataset

AgreeTarget(x_train_sel_with_outliers, y_train)
AgreeTarget(x_test_sel_with_outliers, y_test)
AgreeTarget(x_train_sel_without_outliers, y_train)
AgreeTarget(x_test_sel_without_outliers, y_test)
AgreeTarget(x_train_sel_with_outliers_standarscale, y_train)
AgreeTarget(x_test_sel_with_outliers_standarscale, y_test)
AgreeTarget(x_train_sel_without_outliers_standarscale, y_train)
AgreeTarget(x_test_sel_without_outliers_standarscale, y_test)
AgreeTarget(x_train_sel_with_outliers_mMScale, y_train)
AgreeTarget(x_test_sel_with_outliers_mMScale, y_test)
AgreeTarget(x_train_sel_without_outliers_mMScale, y_train)
AgreeTarget(x_test_sel_without_outliers_mMScale, y_test)

#Para acabar nos guardamos los datasets en un excel
def DataToExcel(dataset, filename):
    return dataset.to_excel(f"../data/processed/{filename}.xlsx")

DataToExcel(x_train_sel_with_outliers, "x_train_sel_with_outliers")
DataToExcel(x_test_sel_with_outliers, "x_test_sel_with_outliers")
DataToExcel(x_train_sel_without_outliers, "x_train_sel_without_outliers")
DataToExcel(x_test_sel_without_outliers, "x_test_sel_without_outliers")
DataToExcel(x_train_sel_with_outliers_standarscale, "x_train_sel_with_outliers_standarscale")
DataToExcel(x_test_sel_with_outliers_standarscale, "x_test_sel_with_outliers_standarscale")
DataToExcel(x_train_sel_without_outliers_standarscale, "x_train_sel_without_outliers_standarscale")
DataToExcel(x_test_sel_without_outliers_standarscale, "x_test_sel_without_outliers_standarscale")
DataToExcel(x_train_sel_with_outliers_mMScale, "x_train_sel_with_outliers_mMScale")
DataToExcel(x_test_sel_with_outliers_mMScale, "x_test_sel_with_outliers_mMScale")
DataToExcel(x_train_sel_without_outliers_mMScale, "x_train_sel_without_outliers_mMScale")
DataToExcel(x_test_sel_without_outliers_mMScale, "x_test_sel_without_outliers_mMScale")