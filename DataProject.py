import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif,chi2, SelectKBest, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



#IMPORTATION DES DONNEES

df = pd.read_csv("weather.csv")
print(" ========================================================================================================== \n Infos sur le jeu de données ")
print("\n")
df.info()                                #variables et leur type --> Il y a 142193 exemples, on repère les variables qui possèdent des valeurs manquantes
print("\n", df.columns)
print("\n", df.head())                         #affichage des premières lignes
print("\n", df.describe())                     #Quelques données de statistique descriptive pour chaque variable : on détecte s'il y a des valeurs aberrantes. Pour MaxTemp, il y a des valeurs aberrantes.

X_data = df.iloc[:, : -1]               #features
y_data = df.iloc[:, -1]                 #target variable            

#EXAMEN DES DONNEES

print(" \n ==========================================================================================================\n Examen des données ")

NombreExemples = df.shape[0]
print(" \n Il y a", NombreExemples, "exemples dans notre dataset.")

plt.figure(figsize = (20,5))                #Visualisation de quelques caractéristiques avec des histogrammes
plt.subplot(1, 5, 1)
df['MinTemp'].hist(bins = 20)
plt.title("Température minimale")
plt.subplot(1, 5, 2)
df["MaxTemp"].hist(bins = 20)
plt.title("Température maximale")
plt.subplot(1, 5, 3)
df['Rainfall'].hist()
plt.title("Pluie")
plt.subplot(1, 5 ,4)
df['Pressure3pm'].hist(bins = 20)
plt.title("Pression à 15h")
plt.subplot(1, 5, 5)
df["Humidity3pm"].hist(bins = 20)
plt.title("Humidité à 15h")


#PREPARATION DES DONNEES

print(" ========================================================================================================== \n Préparation des données " )

Valeurs_manquantes = np.array([X_data.iloc[:, i].isna().sum() for i in range(X_data.shape[1])])          #Tableau 1D qui contient le nombre de valeurs manquantes pour chaque variable
print(" \n Liste contenant les valeurs manquantes pour chaque caractéristique :", Valeurs_manquantes) 

X_data.drop(X_data.iloc[:, Valeurs_manquantes > 47000], axis = 1, inplace = True)           #Boolean Indexing pour supprimer les variables qui ont plus de 47 000 valeurs manquantes (environ 1/3 de données manquantes)
X_data.drop("RISK_MM", axis = 1, inplace = True)                                           #Suppression de la variable RISK_MM
encoder = LabelEncoder()                       
Encodage = [i for i in range(X_data.shape[1]) if type(X_data.iloc[1, i]) == str]           #Avant l'encodage, je remplis les valeurs manquantes dans les colonnes qui ont des valeurs quantitatives
for j in Encodage :
    X_data[X_data.columns[j]].fillna(method="bfill", inplace = True)
    X_data[X_data.columns[j]] = encoder.fit_transform(X_data[X_data.columns[j]])             #Encodage
y_data[:] = encoder.fit_transform(y_data)

for k in range(X_data.shape[1]):                                                        #Pour les variables quantitatives, on remplace les valeurs manquantes par la moyenne de chaque variable 
    if k not in Encodage : 
       X_data[X_data.columns[k]].fillna(X_data[X_data.columns[k]].mean(), inplace = True)


print("\n Corrélation des variables avec la variable de sortie : \n \n", X_data.corrwith(y_data, method = "pearson"))                 #Première visualisation de la corrélation des variables avec variable de sortie

RainToday = X_data["RainToday"].values                             #Je stocke la variable RainToday : celle-ci est fortement corrélée avec la variable de sortie. Elle sera supprimé dans X_data à cause du transformeur VarianceThreshold
RainToday = RainToday.reshape(RainToday.shape[0], 1)
Columns = X_data.columns                                       #Columns est un tableau 1D qui servira plus tard pour reconvertir X_data en Dataframe
selector = SelectKBest(f_classif, k = 14)                      #Selection de caractéristiques avec le transformeur SelectKBest, avec un test chi2
X_data = selector.fit_transform(X_data, y_data)
A = selector.get_support()
Columns = Columns [A == True]                                  #On garde dans Columns les variables qui ont été conservées par SelectKBest

       
print("\n Tableau contenant la variance de chaque variable : \n \n", X_data.var(axis = 0))                                    #Selection de caractéristiques avec le transformeur VarianceThreshold --> les variables qui ont les variances les plus faibles sont éliminées
selector2 = VarianceThreshold(threshold = 30)
X_data = selector2.fit_transform(X_data)
B = selector2.get_support()
Columns = Columns[B == True]                                   #On garde dans Columns les variables qui ont été conservées par VarianceThreshold

X_data = np.concatenate((X_data, RainToday), axis = 1)         #Recuperation de la variable RainToday
scaler = StandardScaler()                                      #Normalisation des données
X_data = scaler.fit_transform(X_data)                          #Rq : X_data est devenu un ND array
print("\n Visualisation rapide du jeu de données après nettoyage et traitement des données : \n \n", X_data)

#Recherche de corrélations

print(" ==========================================================================================================\n Recherche de corrélations \n ")

X_data = pd.DataFrame(X_data, columns = np.concatenate((Columns, np.array(['RainToday']))))                                     #On reconvertit X_data en DataFrame
y_data = pd.Series(y_data, name = "RainTomorrow")                                     #On reconvertit y_data en serie
Z = pd.concat([X_data, y_data], axis = 1)                                             #Concaténation afin de pouvoir utiliser la méthode de corrélation scatter_matrix entre les entrées et la sortie 

print("\n Nouvelle visualisation de la corrélation des variables restantes avec la variable de sortie (vérification de la pertinence de la sélection des variables): \n \n", X_data.corrwith(y_data, method = "pearson"), "\n")
print("\n Corrélation des variables avec la variable MaxTemp : \n \n", X_data.corrwith(X_data["MaxTemp"], method = "pearson"))
print(pd.plotting.scatter_matrix(Z[["Temp3pm","MaxTemp","RainToday","RainTomorrow"]], diagonal = "kde", figsize = (15,5)))


#Extraction des jeux d'apprentissage et de test 

X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size = 0.2)        #Jeux d'apprentissage : 80% du jeu de données
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.scatter(X_train[:1000,0], X_train[:1000,1], c = y_train[:1000], alpha = 0.8)                          #Visualisation de quelques points de MaxTemp en fonction de MinTemp (corrélation forte) pour le jeu d'entraînement et le jeu de test
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.title("Train set")
plt.subplot(1, 2, 2)
plt.scatter(X_test[:1000,0], X_test[:1000,1], c = y_test[:1000], alpha = 0.8)
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.title("Test set")

#Entraînement du modèle 

model = LogisticRegression()                                                                        
model.fit(X_train, y_train)                                                               #Entraînement du modèle
print("\n Score avec le jeu d'entraînement : ", model.score(X_train, y_train))            #Score obtenu sur le jeu d'entraîenement

#Evaluation du modèle 

print("\n ==========================================================================================================\n Evaluation du modèle \n")

y_pred = model.predict(X_test)                                               #test du modèle sur le jeu de test

print("y prédit", "y vrai \n")
for i in range(20):
   print(model.predict(X_test)[i], y_test[i])                               #comparaison pour quelques exemples du jeu de test des valeurs prédites par le modèle et des valeurs vraies 
   
print("\n accuracy score :", accuracy_score(y_test, y_pred))                #Analyse de la performance du modèle avec quelques métriques
print("\n confusion_matrix :", confusion_matrix(y_test, y_pred))
print("\n precision_score :", precision_score(y_test, y_pred))
print("\n recall_score :",recall_score(y_test,y_pred))
print("\n f1_score :",f1_score(y_test, y_pred))

#Amélioration de l'évaluation

print("\n ==========================================================================================================\n Validation croisée \n")

CV = StratifiedKFold(5, shuffle = True, random_state = 0)                                    #Validation croisée à 5-plis stratifiée
Scoring = ['accuracy', 'average_precision', 'recall', 'f1']                                  #Liste de scores 
Results = cross_validate(LogisticRegression(), X_data, y_data, cv = CV, scoring = Scoring )  #On effectue la validation croisée et on récupère les scores obtenus pour chaque fold 
print("\n Résultats avec test sur les différents folds : \n \n ", Results, "\n")             

print("Score moyen avec incertitude : \n")                                                   #On moyenne les scores sur les 5 folds pour obtenir les scores de validation croisée 
for i in Results.keys() :
    print(i, ":", Results[i].mean(), "+/-", Results[i].std())
    
print("\n ========================================================================================================== \n GRAPHIQUES : Histogrammes, Corrélation avec scatter_matrix et nuage de points entre MaxTemp et MinTemp : \n" )    


 