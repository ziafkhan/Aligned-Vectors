from get_glove_embeddings import glove
import numpy as np
import argparse

# from https://github.com/babylonhealth/fastText_multilingual/blob/master/align_your_own.ipynb
# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy

def normalized(a:np.array, axis=-1, order=2) -> np.array:
	"""
	Parameters
	----------
	a : array to be normalized
	"""
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2==0] = 1
	return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary:glove, target_dictionary:glove, bilingual_dictionary:list):
	"""
	Parameters
	----------
	source_dictionary : word vector object containing all words of the source language
	target_dictionary : word vector object containing all words of the target language
	bilingual_dictionary : list of translation pair tuples [(source_word, target_word), ...].
	"""
	source_matrix = []
	target_matrix = []

	for (source, target) in bilingual_dictionary:
		
		try:
			assert source in source_dictionary
			assert target in target_dictionary
		except AssertionError:
			if (source not in source_dictionary) and (target not in target_dictionary):
				print (f"Warning : Couplet not found - {source}, {target}")
			elif target not in target_dictionary:
				print (f"Warning : Target not found - {target}")
			else:
				print (f"Warning : Source not found - {source}")
		source_matrix.append(source_dictionary[source])
		target_matrix.append(target_dictionary[target])

	# return training matrices
	return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix:np.array, target_matrix:np.array, normalize_vectors:bool=True) -> np.array:
	"""
	Parameters
	----------
	source_matrix : shape (dictionary_length, embedding_dimension)
	target_matrix : shape (dictionary_length, embedding_dimension)
	These contain paired word vectors from the bilingual dictionary.

	normalize_vectors : Set to true to normalize the source and target matrices.
	"""
	# optionally normalize the training vectors
	if normalize_vectors:
		source_matrix = normalized(source_matrix)
		target_matrix = normalized(target_matrix)

	# perform the SVD
	product = np.matmul(source_matrix.transpose(), target_matrix)
	U, s, V = np.linalg.svd(product)

	# return orthogonal transformation which aligns source language to the target
	return np.matmul(U, V)

def generate_bilingual_dictionary(use_same_words:bool=True,mapping_dict:str=None,embedding_1:str=None,embedding_2:str=None):
	"""
	use_same_words : Use common words in the dictionary (like words that mean same thing in similar languages) to augment
					 the bilingual dictionary
	mapping_dict   : Dictionary mapping words between the two languages
	embedding_1    : Text file containing word embeddings
	embedding_2    : Text file containing word embeddings  
	"""
	bilingual_dictionary = []

	if mapping_dict is not None:
		with open(mapping_dict, encoding='utf-8') as file:
			for line in file:
				line = line.strip()
				bilingual_dictionary.append((line.split(",")[0],line.split(",")[1]))

	if use_same_words and ((embedding_1 is None) or (embedding_2 is None)):
		raise ValueError("Need embeddings to get common words.")

	if use_same_words:
		common_words = embedding_2.words.intersection(embedding_1.words)
		bilingual_dictionary.extend([(i,i) for i in common_words])

	if len(bilingual_dictionary) == 0:
		raise ValueError("bilingual_dictionary has no entries. Cannot generate transform.")

	return bilingual_dictionary


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Alignment of different word vectors')
	parser.add_argument("-v1", "--vector_1", help = "Word vector (glove type txt file) to which v2 will be aligned", required=True)
	parser.add_argument("-v2", "--vector_2", help = "Word vector (glove type txt file) to be aligned with v1 via transformation", required=True)
	parser.add_argument("-dm", "--dict_mapping", help = "CSV File (v2_word,v1_word) from which same words of different languages may be picked up. No headers.")
	parser.add_argument("-usm", "--use_same_words", action='store_true', help = "use words occuring in both word vectors embeddings as the same word.")
	parser.add_argument("-dv", "--do_val", action='store_true', help = "run validation function in glove class before loading.")
	args = parser.parse_args()

	#Load Embeddings
	embedding_1 = glove(args.vector_1, args.do_val)
	embedding_2 = glove(args.vector_2, args.do_val)

	#Load dict for mapping    
	bilingual_dictionary = generate_bilingual_dictionary(args.use_same_words,args.dict_mapping,embedding_1,embedding_2)
	source_matrix, target_matrix = make_training_matrices(embedding_2, embedding_1, bilingual_dictionary)
	transform = learn_transformation(source_matrix, target_matrix)
	embedding_2.apply_transform(transform)
	print("Transformation complete. Writing update version to file.")
	out_file_name = args.vector_2.replace(".txt","")+"_aligned_"+args.vector_1
	embedding_2.save_to_file(out_file_name)
	embedding_1.save_to_file(out_file_name,"a")