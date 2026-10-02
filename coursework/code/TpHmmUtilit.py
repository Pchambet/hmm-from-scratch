#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Oct 2022
Modified Nov. 2024
@author: barthes
Version 1.4
Attention : il convient d'installer le package : pomegranate > 1.1.X
"""
from scipy.io import wavfile
import numpy as np
from base import mfcc, delta, logfbank, fbank
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import matplotlib.patches as pat
from sklearn.metrics import confusion_matrix
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Support multiple pomegranate versions: try to import a Normal-like distribution
try:
    from pomegranate.distributions import Normal
except Exception:
    try:
        # newer / alternate entrypoints
        from pomegranate import NormalDistribution as Normal
    except Exception:
        from pomegranate.distributions import NormalDistribution as Normal
try:
    from pomegranate.hmm import DenseHMM
except Exception:
    try:
        from pomegranate import HiddenMarkovModel as DenseHMM
    except Exception:
        from pomegranate.hmm import HiddenMarkovModel as DenseHMM
import torch 

class ListeFeat:
    def __init__(self,liste,label,methode,featStart,featStop):
        self.liste=liste
        self.label=label
        self.feature=methode
        self.featStart=featStart
        self.featStop=featStop
    def __getitem__(self,index):
        return self.liste[index]
    def __len__(self):
        return len(self.liste)
    def getListe(self):
        return self.liste
    def getLabel(self):
        return self.label
    def getMethode(self):
        return self.feature
    def getFeatStart(self):
        return self.featStart
    def getFeatStop(self):
        return self.featStop
    def splitListe(self,Itrain,Itest):
        '''
        Coupe une liste en 2 sous listes (une pour l'apprentissage et une pourle test)
        '''
        listeTrain=[item for i,item in enumerate(self.liste) if i in Itrain]
        listeTest=[item for i,item in enumerate(self.liste) if i in Itest]
        train= ListeFeat(listeTrain,self.label,self.feature,self.featStart,self.featStop)
        test= ListeFeat(listeTest,self.label,self.feature,self.featStart,self.featStop)   
        return train,test
        
class Words:
    '''
    Cette classe permet de lire des fichiers audio, de les découper en Frame,
    et de claculer pour chaque Frame un certain nombre de Features
    '''
    def __init__(self,rep='audio',name=None,numcep=13,lowfreq=0,highfreq=None,winlen=0.02,winstep=0.01,nfilt=26,nfft=256,filterLow=False,noise=0):
        data,labels,words,rate = self._loadWaveFiles_(rep,filterLow=filterLow,noise=noise)
        warnings.filterwarnings("ignore")
        self.data = data
        self.labels = labels
        self.words=words
        self.rate = rate
        self.name=name
        self.numcep=numcep
        self.lowfreq=lowfreq
        self.highfreq=highfreq
        self.winlen=winlen
        self.winstep=winstep
        self.nfilt=nfilt
        self.nfft=nfft
        self._extractFeatures_()  
    def __add__(self,other):
        import copy
        tmp=copy.deepcopy(self)
        for i,item in enumerate(other.labels):
            tmp.data.append(other.data[i])
            tmp.labels.append(other.labels[i])
            tmp.rate.append(other.rate[i])
        tmp.name+='-'+other.name
        tmp.words=list(set(self.words+other.words))
        tmp._extractFeatures_() 
        return tmp
    def __len__(self):
        return len(self.labels)
    def getNfilt(self):
        return self.nfilt
    def getNfft(self):
        return self.nfft
    def getNumcept(self):
        return self.numcep
    def getRate(self):
        return self.rate
    def getData(self):
        return self.data
    def getName(self):
        return self.name
    def getWinLen(self):
        return self.winlen
    def getWinStep(self):
        return self.winstep
    def getLowFreq(self):
        return self.lowfreq
    def getHighFreq(self):
        return self.highfreq
    def _loadWaveFiles_(self,path='audio',verbose=True,filterLow=False,noise=0):
        print('Chargement des fichiers audio ...')
        filePaths = []
        labels = []
        words = []
        # If the provided path does not exist relative to the kernel cwd,
        # try a few likely alternatives (project data folders).
        if not os.path.isdir(path):
            # If a file was accidentally passed (e.g. a .DS_Store entry), use its parent directory
            if os.path.isfile(path):
                path = os.path.dirname(path)
            else:
                # Try project data folder relative to this utility file
                candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', path))
                if os.path.isdir(candidate):
                    path = candidate
                else:
                    # Fallback to repository root data folder
                    repo_candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', path))
                    if os.path.isdir(repo_candidate):
                        path = repo_candidate
                    else:
                        # As a last resort, set to candidate (may raise later with clear message)
                        absolute_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', path))
                        path = absolute_fallback

        # Final validation: if path still points to a file, use its parent; if not a directory, raise informative error
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not os.path.isdir(path):
            raise NotADirectoryError(f"_loadWaveFiles_: resolved path is not a directory: {path}")
        # Debug: report resolved path and its entries
        try:
            entries = os.listdir(path)
        except Exception as _e:
            print(f"_loadWaveFiles_: cannot list path '{path}':", _e)
            raise
        print(f"_loadWaveFiles_: using path = {path}")
        print(f"_loadWaveFiles_: entries = {entries}")
        for f in entries:
            fpath = os.path.join(path, f)
            # skip non-directories (such as .DS_Store) and hidden files
            if not os.path.isdir(fpath) or f.startswith('.'):
                continue
            for w in os.listdir(fpath):
                filePaths.append(os.path.join(fpath, w))
                labels.append(f)
                if f not in words:
                    words.append(f)
        data=[]
        rate=[]
        for n,file in enumerate(filePaths):
            rate0, d = wavfile.read(file)
            if noise > 0:
                d=d+np.random.normal(0,1,len(d))* noise          
            
            if filterLow :
                d=filtre(d)
            d =d-np.mean(d)
            #maxi=np.max(np.abs(d))
            #d=d/np.sqrt(np.sum(d**2)/len(d))
            #d=d/maxi
            data.append(d)
            rate.append(rate0)
        if verbose:
            print('Mots trouvés:', words)
            print('Nombre total de mots:',len(labels))
        print('Chargement terminé ...')
        return data,labels,words,rate
    def __str__(self):
        return 'Name : {}, nombre de mots : {}'.format(self.name,len(self.rate))
    def __repr__(self):
        return self.__str__()
    def getLabels(self):
        return self.words
    def __getitem__(self,methode):
        return self.features[methode]
    def getSequenceObs(self,methode='mfcc',recordNumber=0):           # une sequence
        return self.features[methode][recordNumber]
    def getVecteurObs(self,methode='mfcc',recordNumber=0,trameNumber=0):    # une vecteur de la sequence 
        return self.getSequenceObs(methode,recordNumber)[trameNumber]
    
    def _extractFeatures_(self):
        print('Extraction des features ...')
        mfcc_feat=[]    
        filter_feat=[]
        spectrum=[]
        
        for i in range(len(self.rate)):
            mfcc_feat.append(mfcc(self.data[i],self.rate[i],self.winlen,self.winstep,self.numcep,self.nfilt,self.nfft,self.lowfreq,self.highfreq))
            filterFeat,energy,spectrum0 = fbank(self.data[i],self.rate[i],self.winlen,self.winstep,self.nfilt,self.nfft,self.lowfreq,self.highfreq)
            filterFeat=10*np.log10(filterFeat)
            #filterFeat -= (numpy.mean(filterFeat, axis=0) + 1e-8)
            #filterFeat /=numpy.std(filterFeat, axis=0)
            spectrum0=10*np.log10(spectrum0)
            
            #spectrum0 -= (numpy.mean(spectrum0, axis=0) + 1e-8)
            #spectrum0 /=numpy.std(spectrum0, axis=0)
            
            filter_feat.append(filterFeat)
            spectrum.append(spectrum0)
        self.features={'mfcc':mfcc_feat,'filter':filter_feat,'spectrum':spectrum}
        print('Extraction des features terminée ...')
        
    def plotOneWord(self,label='orange',num=0):
        '''
        Affiche le spectrogramme, le spectrogramme filtré et les coefficients de Mel.
        Par défaut on affiche le premier fichier (num=0) du mot orange.
        '''
        index=[i for i,item in enumerate(self.labels) if item==label]
        num=index[0]+num
        plt.figure()
        ax=plt.subplot(4,1,1)      # signal temporel
        plt.plot(np.arange(len(self.data[num]))/self.rate[num],self.data[num], color='blue')
        ax.set_xlim(0, len(self.data[num])/self.rate[num])
        plt.title('Série temporelle de {}[{}]'.format(self.labels[num],num-index[0]),loc='left')
        plt.xlabel('Time (seconde)')
        plt.ylabel('Amplitude (V)')
        
        
        ax=plt.subplot(4,1,2)
        plt.pcolormesh(self.features['spectrum'][num].T,cmap='jet')
        plt.yticks([0,self.features['spectrum'][num].shape[1]], ['0',str(self.rate[num]/2)])
        plt.title('Spectre',loc='left')
        plt.xlabel('Time (secondes)')
        plt.ylabel('Fréquence (Hz)')
        ax.set_xticklabels(ax.get_xticks()*self.winstep)
        cbaxes = inset_axes(ax, width="20%", height="5%", loc=1) 
        plt.colorbar(cax=cbaxes,orientation='horizontal')
        #plt.colorbar()
        
        ax=plt.subplot(4,1,3)
        plt.pcolormesh(self.features['filter'][num].T,cmap='jet')
        plt.xlabel('Time (secondes)')
        plt.ylabel('filter index')
        plt.title('Filtres',loc='left')
        ax.set_xticklabels(ax.get_xticks()*self.winstep)
        cbaxes = inset_axes(ax, width="20%", height="5%", loc=1) 
        plt.colorbar(cax=cbaxes,orientation='horizontal')
        #plt.colorbar()
        
        ax=plt.subplot(4,1,4)
        plt.pcolormesh(self.features['mfcc'][num][:,0:].T,cmap='jet')
        plt.xlabel('Time (secondes)')
        plt.ylabel('Coefficients de Mel index')
        plt.title('mfcc',loc='left')
        ax.set_xticklabels(ax.get_xticks()*self.winstep)
        cbaxes = inset_axes(ax, width="20%", height="5%", loc=1) 
        plt.colorbar(cax=cbaxes,orientation='horizontal')
        #plt.colorbar()
        plt.show(block=False)


    def plotFeatureXY(self,label='apple',methode='all',I=0,J=1):
        '''
        Affiche dans un plan les composantes I et J de feature pour tous les mots ==label
        En rouge le début du mot, en vert le milieu du mot, en bleu la fin du mot
        '''
        index=[i for i,item in enumerate(self.labels) if item==label]
        if methode != 'all':
            fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
            for i in index:
                l=self.features[methode][i].shape[0]
                #plt.plot(self.features[feature][i][:,I],self.features[feature][i][:,J],'b.',markersize=3)
                plt.plot(self.features[methode][i][int(2*l/3):,I],self.features[methode][i][int(2*l/3):,J],'b.',markersize=4)
                plt.plot(self.features[methode][i][int(l/3):int(2*l/3),I],self.features[methode][i][int(l/3):int(2*l/3),J],'g.',markersize=4)
                plt.plot(self.features[methode][i][0:int(l/3),I],self.features[methode][i][0:int(l/3),J],'r.',markersize=4)

                plt.title('Features de {} (méthode :{})'.format(label,methode))
                plt.xlabel('Feat[{}]'.format(I))
                plt.ylabel('Feat[{}]'.format(J))
                plt.show(block=False)
            return fig,ax
        else:
            #fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
            plt.figure()
            li=['spectrum','filter','mfcc']
            for k,methode in enumerate(li):
                plt.subplot(2,2,k+1)
                
                for i in index:
                    l=self.features[methode][i].shape[0]
                    #plt.plot(self.features[feature][i][:,I],self.features[feature][i][:,J],'b.',markersize=3)
                    plt.plot(self.features[methode][i][int(2*l/3):,I],self.features[methode][i][int(2*l/3):,J],'b.',markersize=4)
                    plt.plot(self.features[methode][i][int(l/3):int(2*l/3),I],self.features[methode][i][int(l/3):int(2*l/3),J],'g.',markersize=4)
                    plt.plot(self.features[methode][i][0:int(l/3),I],self.features[methode][i][0:int(l/3),J],'r.',markersize=4)
                    plt.title('Features de {} (méthode :{})'.format(label,methode))
                    plt.xlabel('Feat[{}]'.format(I))
                    plt.ylabel('Feat[{}]'.format(J))
                    plt.show(block=False)
        return      
    

    def CorrFeatures(self,label='apple'):
        
        def Corr(methode):
            DataList = self.features[methode][index[0]:index[-1]+1]
        
            for i,d in enumerate(DataList):
                if i==0:
                    data = d
                else:
                    data = np.concatenate((data,d),axis=0)
            cov = np.corrcoef(data.T)
            sn.heatmap(cov, annot=False, fmt='g',vmin=-1,vmax=1)
            plt.title(label + ' features corr matrix ('+methode +')')
            plt.show(block=False)

        import seaborn as sn
        index=[i for i,item in enumerate(self.labels) if item==label]
        
        plt.figure()
        
        ax=plt.subplot(1,3,1)
        Corr('spectrum')
        ax.axis('equal')
        ax=plt.subplot(1,3,2)
        Corr('filter')
        ax.axis('equal')
        ax=plt.subplot(1,3,3)
        Corr('mfcc')   
        ax.axis('equal')
        plt.show(block=False)
        
    def TSNEFeatures(self,label='apple',perplexity=30):
         
         def computeTSNE(methode):
             DataList = self.features[methode][index[0]:index[-1]+1]
         
             for i,d in enumerate(DataList):
                 l=d.shape[0]
                 I1=int(l/3)
                 I2=int(2*l/3)
                
                 col=['r']*I1 + ['g']*(I2-I1)+ ['b']*(l-I2)
                 
                 if i==0:
                     data = d
                     colorlist = col
                 else:
                     data = np.concatenate((data,d),axis=0)
                     colorlist = colorlist + col
             tsne = TSNE(n_components=2, init="pca", perplexity=perplexity, learning_rate=200, max_iter=1300,verbose=4)
             Y = tsne.fit_transform(data)
             plt.scatter(Y[:, 0], Y[:, 1],c=colorlist)
             plt.title('TSNE méthode {} ({})'.format(methode,label))
             plt.show(block=False)


         from sklearn.manifold import TSNE
         index=[i for i,item in enumerate(self.labels) if item==label]
         plt.figure()
         
         ax=plt.subplot(1,3,1)
         computeTSNE('spectrum')
         
         ax=plt.subplot(1,3,2)
         computeTSNE('filter')
         
         ax=plt.subplot(1,3,3)
         computeTSNE('mfcc')   
         

    def histFeatures(self,label='apple',methode='mfcc',featStart=0,featStop=None,binNumber=30):
        index=[i for i,item in enumerate(self.labels) if item==label]
        plt.figure()
        featNum=featStart
        
        if featStop is None: featStop=self.features[methode][0].shape[1]
        numberPlot=featStop-featStart+1
        nC=4
        nL=int(np.ceil(numberPlot/nC))
        numPlot=1
        
        for featNum in range(featStart,featStop+1):
            flatList1=np.array([])
            flatList2=np.array([])
            flatList3=np.array([])
            for i in index:
                l=self.features[methode][i].shape[0]
                flatList1=np.concatenate((flatList1,self.features[methode][i][0:int(l/3),featNum]))
                flatList2=np.concatenate((flatList2,self.features[methode][i][int(l/3):int(2*l/3),featNum]))
                flatList3=np.concatenate((flatList3,self.features[methode][i][int(2*l/3):,featNum]))
            plt.subplot(nL,nC,numPlot)
            plt.hist(flatList1,binNumber,density=True,color='r',histtype='step')
            plt.hist(flatList2,binNumber,density=True,color='g',histtype='step')
            plt.hist(flatList3,binNumber,density=True,color='b',histtype='step')
            plt.xlabel('Feat[{}]'.format(featNum))
            if numPlot==1:
                    plt.title('Hist. Features de {} (méthode : {})'.format(label,methode))
            numPlot+=1
        plt.show(block=False)
            
    def getFeatList(self,label=None,methode='mfcc',featStart=0,featStop=None,dataset=None):
        '''
        Récupère la liste des caractéristiques pour un mot donné et une caractéristique donnée.
        Si label = None alors tous les mots sont retounés
        '''
        if label is not None:
            liste=[self.Array2List(item,featStart,featStop) for i,item in enumerate(self.features[methode]) if self.labels[i]==label]
        else:
            liste=[self.Array2List(item,featStart,featStop) for i,item in enumerate(self.features[methode])]
        #return {'liste':liste,'label':label,'feature':feature,'featStart':featStart,'featStop':featStop,'size':len(liste)}
        if dataset == 'train':
            return ListeFeat(liste[0:10],label,methode,featStart,featStop)
        elif dataset=='val':
            return ListeFeat(liste[10:],label,methode,featStart,featStop)
        return ListeFeat(liste,label,methode,featStart,featStop)
    
    def Array2List(self,Array,featStart=0,featStop=None):
        return Array[:,featStart:featStop+1]



class GaussianHMM:
    '''
    Cette classe permet de créer et d'entrainer une chaine de Markov cachée à partir d'un ensemble de séquences stockées dans liste.
    L'attribut model contient le modèle.
    L'attribut Nstates contient le nombre d'état de la chaine
    Si model == None alors le modele est recupere en argument
    '''
    def __init__(self,liste=None,Nstates=4,model=None,max_iter=1000,Nsamples=1000):
        if liste is not None:
            
            print('Learning ...')
            
            self.distributions=[]
            start_probability=[]
            ends=[]
            for _ in range(Nstates):
                self.distributions.append(Normal(0, 1))
                start_probability.append(0.0)
                ends.append(0)
            ends[-1]=1e-20
            start_probability[0] = 1.0
            return_states=True
            self.model= DenseHMM(self.distributions, edges=None, starts=start_probability,ends=ends,tol=0.001,
                             sample_length=Nsamples,return_sample_paths=return_states,max_iter=max_iter, verbose=True,inertia=0.0)
            
            liste2=[torch.tensor(l).unsqueeze(0) for l in liste.liste]
            self.model.fit(liste2)
            #self.model.fit(liste['liste'])
            print('Done ...')
        else:
            import copy
            self.model=copy.deepcopy(model)
        self._getTansMuCovMatrix_()
        self.label=liste.getLabel()
        self.feature=liste.getMethode()
        self.featStart=liste.getFeatStart()
        self.featStop=liste.getFeatStop()
        self.Nstates=Nstates
    def __str__(self):
        return 'Model Gaussian HMM\nNstates : {}\nWord : {}\nFeature : {}\nFeat Start : {}\nFeat Stop : {}'.format(
                self.Nstates,self.label,self.feature,self.featStart,self.featStop)
    
    def log_prob(self,liste):
        '''
        Retourne la log densité de probabilité d'observer une séquence sachant le modèle. Si liste contient N
        séquences alors une liste de N probabilités est retournée
        '''
        if type(liste) is np.ndarray:
            return self.model.log_probability(liste)
        listeLogProb=[]
        for i in range(len(liste)):
            tmp=np.reshape(liste[i],(1,-1,len(self.mu[0])))
            listeLogProb.append(self.model.log_probability(tmp))
        
        return listeLogProb
    
    def predict(self,liste):
        '''
        Retourne la séquence d'état la plus probable d'une séquence sachant le modèle. Si liste contient N
        séquences alors une liste de N séquences est retournée
        '''
        if type(liste) is np.ndarray:
            return np.array(self.model.predict(liste))
        
        listeSeqEtats=[]
        for i in range(len(liste)):
            tmp=np.reshape(liste[i],(1,-1,len(self.mu[0])))
            a=self.model.predict(tmp)
            listeSeqEtats.append(np.array(a))
        return listeSeqEtats
    
    def _getTansMuCovMatrix_(self):
        '''
        Récupère les matrices de transitions ainsi que les matrices des lois de probabilité d'émission
        (moyenne mu et covariance cov) et les stocke dans des attributs de meme nom.
        '''
        mu=[]
        cov=[]
        name=[]
        
        for i,d in enumerate(self.distributions):
            it=d.parameters()
            dummy=next(it)
            mu.append(next(it).detach().numpy())
            cov.append(next(it).detach().numpy())
            name.append(str(i))
        
        self.trans=  self.model.edges.exp().numpy()
        self.trans[self.trans< 1e-15]=0
        self.mu=mu
        self.cov=cov
        self.stateName=name
        self.pi0=np.exp(self.model.starts)
        #self.pi0[self.pi0 < 1e-10]=0
    def getPi0(self):
        return self.pi0
    def getMu(self):
        return self.mu
    def getCov(self):
        return self.cov
    def getTrans(self):
        return self.trans
    def plotGaussianConfidenceEllipse(self,words,Fx=0,Fy=1,color='r',zorder=0):
        '''
        Affiche dans un plan les composantes I et J de features pour tous les mots =label
        En rouge le bébut du mot, en vert le milieu du mot, en bleu la fin du mot
        Affiche les ellipses à 95% associées à chacun des états
        '''
        F1,F2 = Fx-self.featStart,Fy-self.featStart
        fig,ax=words.plotFeatureXY(self.label,self.feature,Fx,Fy)
        for i in range(len(self.mu)):
            mean=self.mu[i]
            covariance=self.cov[i]
            plt.plot(mean[F1], mean[F2], "r*", zorder=zorder)
            plt.text(mean[F1]*0.9, mean[F2],self.stateName[i],fontsize=16,color='k',horizontalalignment='center',zorder=2)

            if covariance.ndim == 1:
                covariance = np.diag(covariance)
            
            cov=np.zeros((2,2))
            cov[0,0]= covariance[F1,F1]
            cov[1,1]= covariance[F2,F2]
            cov[0,1]= covariance[F1,F2]
            cov[1,0]= covariance[F2,F1]
            radius = np.sqrt(5.991)     # 95%
    
            eigvals, eigvecs = np.linalg.eig(cov)
            signe=1
            if eigvals[0] > eigvals[1] : 
                nn1=0
                nn2=1    
            else:
                nn1=1
                nn2=0
                signe=-1
            axis = np.sqrt(eigvals) * radius
        #slope = eigvecs[1][n1] / eigvecs[1][n2]
            slope = eigvecs[nn2][nn1] / eigvecs[nn1][nn1]
            angle = signe*180.0 * np.arctan(slope) / np.pi
        
            e=pat.Ellipse((mean[F1],mean[F2]), 2 * axis[0], 2 * axis[1], angle=angle,
                          fill=False, color='red', linewidth=2)
            ax.add_artist(e)
            plt.xlabel('Feat{}'.format(Fx))
            plt.ylabel('Feat{}'.format(Fy))
            #plt.axis([-5,5,-5,5])

    
def filtre(data):
    fc = 0.1
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)
 
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)
    new_signal = np.convolve(data, sinc_func)
    return new_signal

def autocorr(x):
    x2 = x -np.mean(x)
    autocorr_f = np.correlate(x2, x2, mode='full')
    maxi=max(autocorr_f)
    return a