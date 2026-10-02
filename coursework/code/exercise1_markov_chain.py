# -*- coding: utf-8 -*-
"""
Exercice 1 - Chaîne de Markov pluie / sèche
Simulation simple avec NumPy (sans pomegranate)

États :
    0 = "sec"
    1 = "pluie"
"""

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as mio

from duree import duree

#%% Paramètres du modèle de Markov

# Probabilité initiale : P(X0 = sec) = gamma
gamma = 0.95

# Paramètres de transition (à adapter selon l’énoncé si besoin) :
# T[i, j] = P(X_t = j | X_{t-1} = i)
#
# Ici on prend par exemple :
#  - depuis "sec" (0)  : P(sec -> sec)   = alpha,  P(sec -> pluie)  = 1 - alpha
#  - depuis "pluie" (1): P(pluie -> sec) = beta,   P(pluie -> pluie)= 1 - beta
alpha = 0.65
beta  = 0.02

Nsamples = 10000  # longueur de la séquence simulée

# Distribution initiale
start_probability = np.array([gamma, 1 - gamma], dtype=float)

# Matrice de transition
T = np.array([
    [alpha,       1.0 - alpha],   # depuis état 0 ("sec")
    [beta,        1.0 - beta]     # depuis état 1 ("pluie")
], dtype=float)

#%% Simulation de la chaîne de Markov (états = observations)

# 0 = sec, 1 = pluie
statesSeq = np.zeros(Nsamples, dtype=int)
obsSeq    = np.zeros(Nsamples, dtype=int)

# État initial
statesSeq[0] = np.random.choice([0, 1], p=start_probability)
obsSeq[0]    = statesSeq[0]   # obs = état

# Évolution de la chaîne
for t in range(1, Nsamples):
    prev = statesSeq[t - 1]
    statesSeq[t] = np.random.choice([0, 1], p=T[prev])
    obsSeq[t]    = statesSeq[t]

# Fréquence empirique de pluie dans la séquence simulée
freq_pluie_emp = np.mean(obsSeq == 1)
print("Fréquence empirique de pluie (simulée) :", freq_pluie_emp)

#%% Calcul des durées sèches et pluvieuses sur la série simulée

dSec, dPluie, pdfSec, pdfPluie, binsSec, binsPluie = duree(obsSeq)

print("Nombre de périodes sèches (simulées)      :", len(dSec))
print("Nombre de périodes pluvieuses (simulées)  :", len(dPluie))

#%% Étude des périodes pluie / sèche sur la série expérimentale

# On construit un chemin robuste vers ../../data/RR5MN.mat
this_dir = os.path.dirname(__file__)
mat_path = os.path.join(this_dir, "..", "..", "data", "RR5MN.mat")

ObsMesure = mio.loadmat(mat_path)['Support'].astype(np.int8).squeeze() - 1
# ObsMesure est normalement une séquence de 0/1 (sec/pluie)

dSecMes, dPluieMes, pdfSecMes, pdfPluieMes, binsSecMes, binsPluieMes = duree(ObsMesure)

print("Nombre de périodes sèches (mesurées)      :", len(dSecMes))
print("Nombre de périodes pluvieuses (mesurées)  :", len(dPluieMes))

#%% (Optionnel) Tracés comparatifs pour le rapport

if __name__ == "__main__":
    # Les bins sont typiquement des bords -> longueur = len(pdf)+1
    # On prend donc les centres de classes pour l'affichage.
    centersSec     = (binsSec[:-1]     + binsSec[1:])     / 2
    centersSecMes  = (binsSecMes[:-1]  + binsSecMes[1:])  / 2
    centersPluie   = (binsPluie[:-1]   + binsPluie[1:])   / 2
    centersPluieMes= (binsPluieMes[:-1]+ binsPluieMes[1:])/ 2

    # Histogrammes des durées sèches simulées vs mesurées
    plt.figure()
    plt.bar(centersSec, pdfSec, width=1, alpha=0.5, label='Sèche simulée')
    plt.bar(centersSecMes, pdfSecMes, width=1, alpha=0.5, label='Sèche mesurée')
    plt.xlabel("Durée des périodes sèches")
    plt.ylabel("Densité")
    plt.title("Durées des périodes sèches (simulé vs mesuré)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histogrammes des durées pluvieuses simulées vs mesurées
    plt.figure()
    plt.bar(centersPluie, pdfPluie, width=1, alpha=0.5, label='Pluie simulée')
    plt.bar(centersPluieMes, pdfPluieMes, width=1, alpha=0.5, label='Pluie mesurée')
    plt.xlabel("Durée des périodes pluvieuses")
    plt.ylabel("Densité")
    plt.title("Durées des périodes pluvieuses (simulé vs mesuré)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()