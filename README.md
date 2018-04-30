# Zero_shot_action_reco_STGCN

Zero shot action recognition using Spatial Temporal Graph Convolutional Network and Deep Visual-Semantic Embedding Model


To generate embeddings for action class names:

- Download fasttext: (clone the repo)
  https://github.com/facebookresearch/fastText

- Download pretrained skip-gram model
  https://drive.google.com/file/d/0B6VhzidiLvjSaER5YkJUdWdPWU0/view

  More pretrained models available here: 
  https://github.com/epfml/sent2vec


- Navigate to the fasttext folder and run the fasttext binary 
- Provide your downloaded model as a .bin and your sentences as a txt file

  $ ./fasttext print-sentence-vectors model.bin < text.txt
  $ ./fasttext print-sentence-vectors wiki_unigrams.bin < action_classes_original.txt
  This will print embeddings (with classnames) to console
  
  
- You can also pipe the text to an output file 
- still some preprocessing reqd until you can make it a numpy array (will update later)




