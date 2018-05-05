import matplotlib.pyplot as plt
from numpy.random import rand
import numpy as np
import matplotlib.cm as cm


classnames = ['drink water',
 'eat meal / eat snack',
 'brushing teeth',
 'brushing hair',
 'drop',
 'pickup',
 'throw',
 'sitting down',
 'standing up from sitting position',
 'clapping',
 'reading',
 'writing',
 'tear up paper',
 'wear jacket',
 'take off jacket',
 'wear a shoe',
 'take off a shoe',
 'wear on glasses',
 'take off glasses',
 'put on a hat / put on a cap',
 'take off a hat / take off a cap',
 'cheer up',
 'hand waving',
 'kicking something',
 'put something inside pocket / take out something from pocket',
 'hopping / one foot jumping',
 'jump up',
 'make a phone call / answer phone',
 'playing with phone / playing with tablet',
 'typing on a keyboard',
 'pointing to something with finger',
 'taking a selfie',
 'check time from watch',
 'rub two hands together',
 'nod head / bow',
 'shake head',
 'wipe face',
 'salute',
 'put the palms together',
 'cross hands in front / say stop',
 'sneeze / cough',
 'staggering',
 'falling',
 'touch head / headache',
 'touch chest / stomachache / heart pain',
 'touch back / backache',
 'touch neck / neckache',
 'nausea or vomiting condition',
 'use a fan with hand or paper / feeling warm',
 'punching other person / slapping other person',
 'kicking other person',
 'pushing other person',
 'pat on back of other person',
 'point finger at the other person',
 'hugging other person',
 'giving something to other person',
 "touch other person's pocket",
 'handshaking',
 'walking towards each other',
 'walking apart from each other']





#f = np.load('tsne_feat.npy')
f = np.load('tsne_visual_features_train.npy')

x = f[:, 0]
y = f[:, 1]

fig, ax = plt.subplots()
color_array = np.random.rand(60,3)    # first index has number of classes len(classnames)
# for color in ['red', 'green', 'blue']:
#     n = 750
#     x, y = rand(2, n)
#     scale = 200.0 * rand(n)

labels = np.load('sampled_150_random_pretrain_labels.npy')
#labels = np.loadtxt("mnist2500_labels.txt")

# # colors = cm.rainbow(np.linspace(0, 1, 60))
# for i,label in enumerate(labels):
# 	ax.scatter(x[i],y[i],c=color_array[int(label),:],label=label)

for l in np.unique(labels):
	ix = np.where(labels == l)
	ax.scatter(x[ix],y[ix],c = color_array[int(l),:],label=classnames[int(l)])


# ax.scatter(x, y, c=colors, #label=classnames,
           # alpha=0.3, edgecolors='none')

ax.legend()
ax.grid(True)

plt.show()








