import os,sys
import numpy as np
import cv2
import collections
import zipfile
from zipfile import ZipFile
from Utils.utils import clear_folder
import gzip
import urllib.request
import requests
import pickle
import subprocess

Datasets = collections.namedtuple('Datasets', ['training', 'testing','validation'])

class Dataset():
	def __init__(self,images, gt = 'superlabel'):
		self.instances = images
		self.num_of_images = len(images)
		self.images_path = os.path.dirname(__file__)+'/Images' 
		self.ground_truth = gt
		self.index = 0

	def next_batch(self, batch_size):
		if batch_size > self.num_of_images:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.index
		self.index += batch_size

		if self.index > self.num_of_images:
			np.random.shuffle(self.instances)
			# Shuffle the data
			self.index = batch_size
			start = 0

		end = self.index
		imgs = self.instances[start:end]
		imagesBatch = []
		labelsBatch = []
		for img in imgs:
			if img.endswith('.png'):
				try:
					image = cv2.imread(self.images_path+'/input/'+img)
					label = np.load(self.images_path+'/'+self.ground_truth+'/'+img.replace('png','npy'))
				except:
					continue
				imagesBatch.append(image)
				labelsBatch.append(label)
		return np.array((imagesBatch,labelsBatch))

class DataHandler:
	def __init__(self):
		self.path = os.path.dirname(os.path.relpath(__file__))

	def build_datasets(self):
		images_path = self.path + '/Images'
		attempt_download_and_or_extraction = False
		data_ready = False
		if os.path.exists(self.path+'/Images'):
			try:
				if len(os.listdir(images_path+'/input')) != len(os.listdir(images_path+'/superlabel')):
					clear_folder(images_path)
					attempt_download_and_or_extraction = True
				else:
					data_ready = True
			except:
				clear_folder(images_path)
				attempt_download_and_or_extraction = True
		else:
			attempt_download_and_or_extraction = True

		if attempt_download_and_or_extraction:
			zip_ready =self.__maybe_download_file_from_google_drive()
			if zip_ready:
				print('Extracting Images Into Images Folder')
				data_ready = self.__extract_images()  
				if data_ready:
					print('\nImage Extraction Completed')
				else:
					print('Image Extraction Incompleted')
		if data_ready:
			zip_file = self.path+'/Images.zip'
			if os.path.exists(zip_file):
				try:
					os.remove(zip_file)
					print('Images.zip was removed')
				except:
					pass

			dataset_pickle_path = self.path+"/dataset.pickle"
			if not os.path.exists(dataset_pickle_path):
				keys = os.listdir(images_path+'/input')
				np.random.shuffle(keys)
				sz = len(keys)
				train_idx = int(sz*0.7)
				test_idx = int(sz*0.95)
				dset = {'training':keys[:train_idx]}
				dset.update({'testing':keys[train_idx:test_idx]})
				dset.update( {'validation':keys[test_idx:]})
				pickle.dump(dset,open(dataset_pickle_path,"wb"))
			else:
				dset = pickle.load(open(dataset_pickle_path,'rb'))

			return Datasets(training=Dataset(dset['training']),
					testing=Dataset(dset['testing']),
					validation=Dataset(dset['validation']))
 
	def __extract_images(self):
		"""Extract the first file enclosed in a zip file as a list of words."""
		cnt = 0
		with ZipFile(self.path+'/Images.zip') as z:
			for member in z.filelist:
				try:
					print('extracting',member.filename,end='\r')
					z.extract(member,path=self.path+'/Images')
				except zipfile.error as e:
					return False
			return True
		
	def __maybe_download_file_from_google_drive(self):
		destination = self.path+'/Images.zip'
		response = None
		if os.path.exists(destination):
			return True
		else:
			# Download zip file containing the dataset images (train,test,validate)
			gid = input('enter google drive id of images -> ')
			subprocess.run(['gdrive','download','--path',self.path,gid])
			for zipfile in os.scandir(self.path):
				if zipfile.name.endswith('zip'):
					os.rename(zipfile.path,zipfile.path.replace(zipfile.name,'Images.zip'))
					break
		return os.path.exists(self.path+'/Images')

if __name__ == '__main__':
	DataHandler()
