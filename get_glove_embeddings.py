import numpy as np
import sys
import os
import warnings
from scipy import spatial
from tqdm import tqdm

class glove():
	"""
	Class to manipulate Glove vectors in txt files.
	At initialization requires glove_path, the path to a file formatted as --

	word v1 v2 ..... vn
	
	where v1 v2 ..... vn are n floats signifying an n-dimensional vector.
	word cannot have spaces.
	To speed up loading, load with validate = False.

	self.glove_dimensionality   - number of dimensions of word vector
	self.glove_file_lines       - number of lines in the word vector file. Used in tqdm.
	self.number_of_words        - total number of entries in the embedding vector
	self.word_to_num            - dictionary containing words as keys and a corresponding
								  unique number as value
	self.num_to_word            - reversed self.word_to_num
	self.glove_embedding_vector - embedding vector
	self.words                  - set of all words in the embedding vector
	"""

	def __init__(self, glove_path:str, validate:bool=True):
		self.glove_path = glove_path

		with open(self.glove_path,'r') as glove_file:
			self.glove_dimensionality = len(glove_file.readline().split())-1 #first item is the word
		self.glove_file_lines = sum(1 for i in open(self.glove_path,'rb')) 
		self.number_of_words = self.glove_file_lines + 2 #adding a plus 2 for future padding and OOV words
		
		with open(self.glove_path,'r') as glove_file:
			self.glove_dimensionality = len(glove_file.readline().split())-1 #first item is the word

		if validate:
			self.__validate_word_embeddings()
			#Have to redo everything incase errors were found.
			self.glove_file_lines = sum(1 for i in open(self.glove_path,'rb')) 
			self.number_of_words = self.glove_file_lines + 2 #adding a plus 2 for future padding and OOV words
			with open(self.glove_path,'r') as glove_file: 
				self.glove_dimensionality = len(glove_file.readline().split())-1 #first item is the word
		
		self.word_to_num = {} #May be used as a part of a tokenizing dict as long as the text file is unchanged
		self.num_to_word = {} #May be used as a part of a tokenizing dict as long as the text file is unchanged
		self.glove_embedding_vector = np.zeros((self.number_of_words,self.glove_dimensionality))
		self.__load_word_embeddings()
		self.words = set(self.word_to_num.keys())


	def __validate_word_embeddings(self):
		"""
		Verify that all the words in the text file are in the first position and are not weird.
		"""
		position = 0
		errors = False
		path = self.glove_path
		print ("Validating...")
		with open(path,'r') as embedding_file:
			for line in tqdm(embedding_file,total=self.glove_file_lines):
				word = line.split()[0]
				if type(word) is str and len(line.split())==self.glove_dimensionality+1:
					position+=1
					continue
				else:
					print ("Something weird found at line {}.".format(position))
					position+=1
					errors = True
					continue
		if errors:
			do_correction = input("Errors found. Create a new file removing troublesome lines? (Y/N) - ")
			if do_correction == "Y":
				print ("Transferring valid embeddings to new file...")
				self.__create_corrected_file()
				print ("Loading Embeddings now...")
			else:
				sys.exit(0)
		else:
			print ("Validation Successful. Loading word embeddings...")


	def __create_corrected_file(self):
		"""
		Create a new file in which incorrect lines are removed.
		"""
		new_file_path = self.glove_path+"_2"
		path=self.glove_path
		lines_removed = 0
		with open(path,'r') as embedding_file:
			with open(new_file_path,'w') as new_embedding_file:
				for line in tqdm(embedding_file, total=self.glove_file_lines):
					word = line.split()[0]
					if type(word) is str and len(line.split())==self.glove_dimensionality+1:
						new_embedding_file.write(line.strip()+"\n")
					else:
						lines_removed +=1
						continue
		print ("Removed {} lines.".format(lines_removed))
		delete_old_file = input("Overwrite previous file? (Y/N) - ")
		if delete_old_file.lower() == "y":
			os.remove(self.glove_path)
			os.rename(new_file_path,self.glove_path)
			print ("File updated successfully.")
		else:
			self.glove_path = new_file_path
			print ("Created new file, original file intact. Updated self.glove_path to {}".format(self.glove_path))

		
	def __load_word_embeddings(self):
		"""
		Create the embedding array. The position 0 is reserved 
		for padding (a vector of 0s).

		The final position is for OOV words. We are not differentiating 
		between OOV words, i.e., all OOV words have the same n-dimensional 
		representation within the embedding matrix.

		However, __getitem__ returns a new random vector for every call, 
		so if we use this class to create a new embedding vector, all 
		words will have their own representation.
		"""
		path=self.glove_path
		with open(path,'r') as embedding_file:
			vector_loc = 1 # position 0 is padding
			for line in tqdm(embedding_file,total = self.glove_file_lines):
				self.word_to_num[line.split()[0].lower()] = vector_loc #line.split()[0] is just the first word of the line , which is the actual word
				self.num_to_word[vector_loc] = line.split()[0].lower()
				try:
					self.glove_embedding_vector[vector_loc,:] = np.array([float(i) for i in line.split()[1:]])
				except ValueError:
					print (line)
					print ("Please check/remove this line in the word vector text file before continuing. Try running with validate = True.")
					sys.exit(0)
				vector_loc+=1
			#Add padding and OOV definitions
			self.glove_embedding_vector[vector_loc,:] = np.random.randn(self.glove_dimensionality)
			self.word_to_num["OOV"] = vector_loc
			self.num_to_word[vector_loc] = "OOV"
			self.word_to_num["PADDING"] = 0
			self.num_to_word[0] = "PADDING"
		print ("Successfully Loaded embedding file.")


	def apply_transform(self, transform):
		"""
		Apply the given transformation to the vector space
		Right-multiplies given transform with embeddings E:
			E = E * transform
		Transform is a numpy ndarray.
		This is primarily used during alignment of word vectors with other languages
		"""
		self.glove_embedding_vector = np.matmul(self.glove_embedding_vector, transform)
		#self.glove_embedding_vector[0] = np.zeros(self.glove_dimensionality)

	
	def similarity(self, word_1, word_2):
		"""
		Returns the cosine similarity between two words.
		word_1 - type - str
		word_2 - type - str
		"""
		if len(set([word_1.lower(),word_2.lower()]) - self.words)>0:
			raise KeyError("Atleast one of the words missing in word embeddings")
		return 1 - spatial.distance.cosine(
			self.glove_embedding_vector[self.word_to_num[word_1],:],
			self.glove_embedding_vector[self.word_to_num[word_2],:])

	
	def most_similar(self, word, count=10):
		"""
		Returns the "count" most similar words to an input "word" 
		based on cosine distance between the vectors.
		word - type - str
		count - type - int
		"""
		assert type(word) is str,"Item needs to be type str, found {}".format(type(word))
		if word not in self.words:
			raise KeyError("Word - {} not found in word embeddings".format(word))

		#https://stackoverflow.com/questions/53455909/python-optimized-most-cosine-similar-vector
		target    = self.glove_embedding_vector[self.word_to_num[word],:].reshape(1,self.glove_dimensionality)
		distances = spatial.distance.cdist(target, self.glove_embedding_vector, "cosine")[0]  
		indices   = np.argpartition(distances,count)[:count+1] #the first word will always be the word itself
		sorted_indices = indices[np.argsort(distances[indices])]
		return_list = []
		for index in sorted_indices:
			return_list.append((self.num_to_word[index],1-distances[index]))
		return return_list[1:]


	def save_to_file(self,file_name=None):
		"""
		Saves current embedding vector in glove file format.
		Does not include the PADDING and the OOV vectors.
		"""

		if file_name is None or file_name==self.glove_path:
			print("Overwriting original file")
			file_name = self.glove_path
		with open(file_name,"w") as file:
			for num in tqdm(range(self.number_of_words-2)):
				num = num+1
				word = self.num_to_word[num]
				vector = self.glove_embedding_vector[num,:]

				text = np.array2string(vector,max_line_width=9999999,formatter={'float':lambda x: "%.5f" % x})[1:-1]
				text = word+" "+text
				file.write(text+"\n")


	def __contains__(self, index):
		return index in self.words
	

	def __getitem__(self,index):
		assert type(index) is str,"Item needs to be type str, found {}".format(type(index))
		if index in self.words:
			return self.glove_embedding_vector[self.word_to_num[index],:]
		warnings.warn("Word {} not found. Returning a random normal vector.".format(index))
		return np.random.randn(self.glove_dimensionality)


	def __len__(self):
		return len(self.words)