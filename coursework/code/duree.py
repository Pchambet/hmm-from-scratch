# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:35:25 2018

@author: barthes
"""
import numpy as np

def duree(sequence):
    sequence=sequence.squeeze()
    dif = np.diff(sequence)
    Imon = np.where(dif==1)[0]
    Idesc = np.where(dif==-1)[0]
    if len(Imon) == 0 or len(Idesc) ==0:
        raise('probleme sequence')
        
    if Idesc[0] < Imon[0]:
        Idesc=Idesc[1:]
    I=min(len(Idesc),len(Imon))
    Idesc=Idesc[:I]
    Imon=Imon[:I]
    DureePluie=Idesc-Imon
    DureeSec=Imon[1:]-Idesc[:-1]
    pdfPluie,binsPluie=np.histogram(DureePluie, bins=np.arange(60), density=True)
    binsSec=np.concatenate((np.arange(10),np.logspace(1,3.69897,100)))
    pdfSec,binsSec=np.histogram(DureeSec, bins=binsSec, density=True)
    return DureeSec,DureePluie,pdfSec,pdfPluie,binsSec,binsPluie
    