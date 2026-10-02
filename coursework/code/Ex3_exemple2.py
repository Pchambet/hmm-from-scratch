#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4  2020
Modifié 28/10/2024, 20/11/2025
@author: barthes
Nécessite pomegranate > 1.X.X
Version 1.3
Script exemple permettant l'apprentissage d'un mot à partir d'une chaine
de Markov cachée à l'aide d'une des 3 méthodes de calcul des features 
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from TpHmmUtilit import GaussianHMM, Words
from sklearn.metrics import confusion_matrix
#%% Ne pas modifier
winlen=0.02         # taille d'une frame en seconde
winstep=0.01        # decalage temporel entre une frame et la suivante en seconde
highfreq=4000       # frequence max en Hz
lowfreq=0           # fréquence min en Hz
nfilt=26            # Nonbre de filtres calculés (methode filter)
numcep=12           # Nombre de coefficients de Mel (methode mfcc)
nfft=256            # Nombre de points pour le calcul du spectre (methode spectrum)
#%%
methode='mfcc'      # Choix de la feature (spectrum, filter ou mfcc)
myWord='apple'       # mot choisi 
featStart=0         # Choix de la composante min de feature
featStop=2          # Choix de la composante max de feature
Nstates=3         # Nombre d'état de la chaine de Markov

#%% lecture des fichiers audio et calcul des features. On bruite légérement les enregistrements
words=Words(rep='fruits',name='fruits',numcep=numcep,lowfreq=lowfreq,
            highfreq=None,winlen=winlen,winstep=winstep,nfilt=nfilt,nfft=nfft,noise=50)  

# On extrait une liste avec les 15 enregistrements du mot défini dans myWord
#en utilisant la méthode definie par la variable methode
liste=words.getFeatList(label=myWord,methode=methode,featStart=featStart,featStop=featStop)  

#%% On crée et on entraine une HMM avec la liste précédente (composée 
# de 15 enregistrements de apple)
Model=GaussianHMM(liste=liste, Nstates=Nstates)    # création et entrainement du modèle 

#%% Affiche tous les individus du mot apple dans un plan Fx, Fy ainsi que 
#les ellipes à 95% des gaussiennes associées à chaque états
   
Model.plotGaussianConfidenceEllipse(words,Fx=0,Fy=1,color='b')     # affichage composantes 0 et 1
Model.plotGaussianConfidenceEllipse(words,Fx=0,Fy=2,color='b')      ## affichage composantes 0 et 2
Model.plotGaussianConfidenceEllipse(words,Fx=1,Fy=2,color='b') 

#%% Visualisation des parametres de la chaine de Markov
np.set_printoptions(precision=2,floatmode='fixed',suppress=False)
print('Matrice de transition :\n{}'.format(Model.getTrans()))
print('Prob initiale : \n{}'.format(Model.getPi0()))
for i in range(Nstates):
    print('\nEtats {} :'.format(i))
    print('cov:\n{}'.format(Model.getCov()[i]))    
    print('Mu:\n{}'.format(Model.getMu()[i]))

#%% prediction des séquence d'état optimale par l'algorithme de 
# Viterbi pour chacun des 15 enregistrements du mot choisi (Apple)
predictedStates=Model.predict(liste)       
for i,l in enumerate(predictedStates):
    print('Séquence des Etats optimaux enregistrement {} de {} :\n {}'.format(i,myWord, l))
   

#%% On calcule la log probabilité pour chacun des 15 enregistrements de Apple
# de apple par l'algorithme Forward
logprobs=Model.log_prob(liste)      
print('Log  de la probabilité des {} enregistrements de {}:\n{}\n'.format(len(liste),myWord,logprobs))
   
#%% On calcule les probabilité des séquences de mots kiwi avec la chaine de MArkov Apple
# otherWord = 'kiwi'
# liste2=words.getFeatList(label=otherWord,methode=methode,featStart=featStart,featStop=featStop)  
# logprobs=Model.log_prob(liste2)      
# print('Log  de la probabilité des {} enregistrements de {}:\n{}\n'.format(len(liste),myWord,logprobs))
plt.show()