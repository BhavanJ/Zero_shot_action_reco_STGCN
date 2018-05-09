# Zero_shot_action_reco_STGCN

Zero shot action recognition using Spatial Temporal Graph Convolutional Network and Deep Visual-Semantic Embedding Model


To generate embeddings for action class names:

- Download fasttext: (clone the repo)
      https://github.com/facebookresearch/fastText

- Download pretrained skip-gram model
      https://drive.google.com/file/d/0B6VhzidiLvjSaER5YkJUdWdPWU0/view

  More pretrained models available here: 
      https://github.com/epfml/sent2vec


- Navigate to the fasttext folder and run the fasttext binary.
      Provide your downloaded model as a .bin and your sentences as a txt file

      $ ./fasttext print-sentence-vectors model.bin < text.txt > outfile.txt
      $ ./fasttext print-sentence-vectors wiki_bigrams.bin < action_classes_original.txt > output_embeddings.txt

      
  This will save embeddings to the output txt file
  
- Run generate_embeddings() in neighbours.py to get these embeddings as a .npy
  
  
  
  

UNSEEN CLASSES 

Hand picked   = [3,8,11,16,51]

NEAREST NEIGH = [14,16,43,49,58]

FURTHER NEIGH = [10,24,41,52,55]

Note: The classes go from  0 to 59



