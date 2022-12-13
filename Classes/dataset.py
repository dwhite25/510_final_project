import csv
import glob
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

''' the gan needs one batch of data at a time, nothing else. the data needs to be:
    - self.objects

    nothing else need be stored in memory at any given time, save perhaps for a list of the pathways to each
    file we'd like to include in the dataset (and perhaps a similar list for the label info, if we decide to 
    store it separately)

    when a call is made to the class from the gan, that's when the dataset class should batch data. 
'''
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
class Dataset():    
    # -----------------------------------------------------------------------------------------------
    def __init__(self, batch_size = 32, pathway = '', mdpath='', outpath = '', endres = 128):
        self.labels         = ['freq', 'amp'] 
        self.name           = 'dataset'
        self.batch_size     = batch_size
        self.batchlist      = []
        self.objectspath    = pathway       # the location of the directory to crawl for all usable objects
        self.outpath        = outpath       # place to save outbound images (will create dir if none exists)
        self.objectslist    = []            # list of pathways to all usable objects within objectspath and its subdirs
        self.nobjects       = 0
        self.endres         = endres        # == the highest resolution of the inbound/outbound time series data
        self.nchannels      = 2             # time and y(t)     
        self.labels_batch   = []            # batch of labels that corresponds 1:1 to batch of time series sims fed into the network
        self.objects        = []            # shape = (B,R,c): bsize, endres, channels
        self.outputs        = []            # images that will be processed as outbound images
        self.mdat           = []            # eventual holder for all metadata info, as pulled from the .csv file

        assert mdpath != '', 'need to pass location of metadata file'
        assert pathway != '', 'need to pass location of folder that contains simulation files'

        self.compile_data(pathway, mdpath)

    # -----------------------------------------------------------------------------------------------
    def compile_data(self, pathway='', mdpath=''):
        self.objectspath = pathway
        self.objectslist = self.get_object_list_from_path()
        self.nobjects    = len(self.objectslist)
        # we want to save metadata file in memory for the entire run here, as it will be needed frequently
        self.mdat        = pd.read_csv(mdpath, sep='\t')
        print('Objects found: ', self.nobjects)

    # -----------------------------------------------------------------------------------------------
    def get_object_list_from_path(self):
        objectslist = glob.glob(self.objectspath + '/*.csv', recursive=True)

        return objectslist

    # -----------------------------------------------------------------------------------------------
    # creates batch of self.batch_size time series objects and batch of self.batch-size corresponding
    def create_batch(self):
        # get rid of old batch and make new one
        self.purge_batch()
        # currently just picking the object randomly from the list, rather than using a queue
        self.batchlist = self.mdat['sim'].sample(n=self.batch_size).to_numpy()
        for path in self.batchlist:
            self.add_object_to_batch(path)
            self.add_labels_to_batch(path)
    
    # -----------------------------------------------------------------------------------------------
    # final object array should be [[full res],[downsample_1],[downsample_2],...,[downsample_n]]
    # where each downsample is 1/2 the resolution of the previous.
    def create_object(self, path):
        x = pd.read_csv('/'+path, usecols=['wave'], sep='\t')
        x = x.to_numpy(dtype=np.float32)
        obj = []

        for i in range(0,9):
            obj.append(np.array(x[::2**i], dtype=np.float32))

        obj = np.array(obj)

        return obj

    # -----------------------------------------------------------------------------------------------
    # adds one time series object to batch
    def add_object_to_batch(self, path):
        obj = self.create_object(path)
        self.objects.append(obj)

    # -----------------------------------------------------------------------------------------------
    # similar to create_object, but for labels. can work with either create_object or add_labels_to_batch
    def create_labels(self, path):
        # pull from the metadata file to find each file's path
        lbls = self.mdat.loc[self.mdat['sim']==path, self.labels]
        lbls = lbls.to_numpy(dtype=np.float32)[0]

        return lbls

    # -----------------------------------------------------------------------------------------------
    # adds one set of labels, corresponding with one time series object, to the batch of labels
    def add_labels_to_batch(self, path):
        lbls = self.create_labels(path)
        self.labels_batch.append(lbls)

    # -----------------------------------------------------------------------------------------------
    # saves copies of time series data coming out of the GAN as a .csv file
    def save_objects(self, batch, lbls, epoch, location):
        lblstr = str('amp_' + str(lbls[1]) + '_freq_' + str(lbls[0]))

        s1 = str(location + '/series')
        s2 = str(s1 + '/' + str(epoch))
        s3 = str(s2 + '/' + str(lblstr))

        Path(s1).mkdir(exist_ok=True)
        Path(s2).mkdir(exist_ok=True)
        Path(s3).mkdir(exist_ok=True)

        i = 1
        for obj in batch:
            obj = np.array(obj)

            np.savetxt(s3 + "/%02d.csv" %(i), obj, delimiter='\t')
            i += 1


    # -----------------------------------------------------------------------------------------------
    # takes the timestamps from a random sim, since timestamps weren't used in the NN
    # then creates an image stack of the amplitude and strain of a sim created in the NN
    def report_objects(self, batch, epoch, dim=0, cols=1, rows=10, normalized=True):
        pathway = self.objectspath + '/1.csv'
        times   = []
        times   = pd.read_csv(pathway, usecols=['time'], sep='\t')
        times   = np.array(times, dtype=np.float32)
        fig, axs = plt.subplots(rows, cols, figsize=(24,24))
        for obj in batch:
            obj2 = np.array(obj)
        for i in range(10):
            axs[i].plot(times, batch[i][:])

        Path(self.outpath).mkdir(exist_ok=True)
        plt.savefig(self.outpath + "/epoch%05d.png" % epoch)
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------
    # empty out all data to get ready for new batch
    def purge_batch(self):
        self.objects        = []
        self.batchlist      = []
        self.labels_batch   = []