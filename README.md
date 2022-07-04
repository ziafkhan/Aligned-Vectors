# Aligned-Vectors

Align disjoint static word vectors using SVD.

  

Though this was initially used to align Glove vectors, we can use it to align any static word embeddings.

  

## Alignment

The code is based on the paper [Offline bilingual word vectors, orthogonal transformations and the inverted softmax.](https://arxiv.org/pdf/1702.03859.pdf) As stated in the paper, "two pre-trained embeddings are aligned with a linear transformation, using dictionaries compiled from expert knowledge."

## Data
Multilingual vectors can be aligned using this code, the example in the repo uses the word vector files located [here.](https://drive.google.com/file/d/17Hl47L84UE4dQdRQJXTKpuQY8o1R9Hf7/view?usp=sharing) The alignment requires a mapping of words in one language to another. A small mapping for hindi to english is located in hin_eng_map.csv.

## Running the code
1. `git clone https://github.com/ziafkhan/Aligned-Vectors.git`
2. `cd Aligned-Vectors`
3. Download the files mentioned in Data section above and unzip them in this folder.
4. `python .\align_word_vectors.py -v1 ft.en.300.txt -v2 ft.hi.300.txt --dict_mapping hin_eng_map.csv`
5. You will have a new set of vectors in a new file in the directory.

Further examples can be found in the notebook `Auxiliary_Useful_Code.ipynb`.

## Results
The similarity of words after the alignment increases considerably, and a larger dictionary would further improve the quality of the alignment. Here are some words in English and Hindi that mean the same thing and their similarities before and after alignment.

|English|Hindi|Pre Alignment Similarity Score|Post Alignment Similarity Score|
|---|---|---|---|
|`hyderabad`|`हैदराबाद`|0.0841|0.5746|
|`pink` |`गुलाबी`|0.0024|0.5959|
|`thief`|`चोर`|-0.0679|0.6603|
|`clothes`|`वस्त्र`|-0.1182|0.4947|
|`wolf`|`भेड़िया`|-0.0332|0.3530|
|`cricket`|`क्रिकेट`|0.0968|0.5452|
|`israel`|`इज़राइल`|-0.0208|0.4426|

The bottom 4 words were not in the alignment dictionary mappings, yet we see a considerable increase in the similarity of these words.
