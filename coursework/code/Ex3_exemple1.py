#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: barthes
Version 1.4
revision nov 2024
revision nov 2025
Script permettant de charger les fichiers audio et de calculer les features de chaque trame à l'aide des 3 méthodes disponibles (spectrum, filter et mfcc)
Affichage de la série temporelle d'un mot + différentes repésentations des features pour les 3 méthodes 
"""
#%%
from TpHmmUtilit import Words
import matplotlib.pyplot as plt

#%%
winlen=0.02         # taille d'une frame en seconde
winstep=0.01        # decalage temporel entre une frame et la suivante en seconde
nfft=256            # Méthode Spectrum : Nombre de points pour le calcul de la FFT => spectre nfft/2 + 1 valeurs
nfilt=26            # Méthode filter : Nonbre de filtres calculés 
numcep=12           # Méthode mfcc : Nombre de coefficients de Mel  
dossierAudio = 'fruits'

#%% Chargement de tous les fichiers audio contenu dans le dossier 'audio' et calcul des features
words=Words(rep=dossierAudio,name='exemple 1',numcep=numcep,winlen=winlen,winstep=winstep,nfilt=nfilt,nfft=nfft,filterLow=False,noise=0)  

# On récupère la liste des mots disponibles et on l'affiche
listeDesMotsDisponibles = words.getLabels()     
print('Les mots disponibles dans le dossier {} sont :\n{}'.format(dossierAudio,listeDesMotsDisponibles))

#%% On choit un des mots
myWord='banana'      # mot choisi parmi 'apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'

#%%
# Affiche l'enregistrement d'un mot et ses features pour les 3 méthodes d'extraction
record = 0       # Selectionne pour le mot choisi l'enregistrement parmi les 15 (0 .. 14)
words.plotOneWord(label=myWord,num=record)    


#%% Pour les 3 méthodes affiche les matrices de corrélation des features
words.CorrFeatures(label=myWord)

#%%
# Pour les 3 méthodes d'extraction on affiche les histogrammes des différentes features Fi du mot choisi (all records)

words.histFeatures(myWord,'spectrum',0,63)        # Features 0 -> 63 de la méthode spectrum
words.histFeatures(myWord,'spectrum',64,128)      # Features 64->128 de la méthode spectrum

words.histFeatures(myWord,'filter',0,25)          # Features 0 -> 25 de la méthode filtre
words.histFeatures(myWord,'mfcc',0,11)            # Features 0-> 11 de la méthode mfcc

#%%
# Pour les 3 méthodes d'extraction on affiche les features Fx et Fy du mot dans le plan X, Y (all records)

X=0                # Feature X en abcisse pour affichage
Y=1                # Feature Y en ordonnée pour affichage

words.plotFeatureXY(myWord,methode='all',I=X,J=Y)   # Affiche pour tous les mots de myWord les features Fx et Fy pour les 3 méthodes
#%% Pour les 3 méthodes d'extraction on représente la TSNE des features (all records) de myWord
words.TSNEFeatures(label=myWord)
plt.show()    
