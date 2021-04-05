import sys
from pathlib import Path

import gensim
import logging
import pprint
import gensim.downloader as api
import numpy as np


def cal_cos_similarity(q, d):
    return np.dot(q, d) / np.linalg.norm(q) * np.linalg.norm(d)


n2 = 20
# path = Path(f"{Path.home()}/datasets/SIGVerse/trial3_phrase/noisy_label/label_{n2}.csv")
# class_list = np.loadtxt(f"{Path.home()}/datasets/SIGVerse/trial3_phrase/noisy_class_list/class_list.txt", delimiter=',', dtype=str)[1:]

print("Load")
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print("Load Completed")
print(model.similarity("Inside", "Outside"))
sys.exit()
l = [t.replace("\"", "").replace("\'", "").replace(",", "") for t in model.index2word]

# with open("t2.txt", "w") as f:
#     for i in l:
#         f.write(f"{i}\n")

# known_obejct = ["Cushion"]
# known_obejct = ["Door", "Clock", "Cushion"]
known_obejct = ["Toilet", "Clock", "Door"]

vecs = []
v = []
for i, c in enumerate(class_list):
    if c == "TrashBox":
        c = "box"
    if c == "StuffedAnimal":
        c = "Stuffed"
    if c == "TissueBox":
        c = "Tissue"
    c = c.capitalize()
    x = model.word_vec(c)
    vecs.append((c, x))
    v.append(c)

array = np.zeros(shape=(len(vecs)))

# results = model.most_similar(positive=v)
# for result in results:
#     print(result)

for i, (name1, v1) in enumerate(vecs):
    sims = 0
    for k, name2 in enumerate(known_obejct):
        sim = model.similarity(name1, name2)
        sims += sim
        # print(name1, name2, sim)
        # array[i, k] = sim
        # print(name1, name2, cal_cos_similarity(v1, v2))
        sims = sims / len(known_obejct)

    array[i] = sims if sims > 0 else 0
    print(sims)

np.savetxt("array_known.csv", array, fmt='%.5f')

# from sklearn.cluster import KMeans
#
# kmeans_model = KMeans(n_clusters=2, random_state=10).fit(array)
# labels = kmeans_model.labels_
# np.savetxt("kmeans.csv", np.array(labels), fmt='%.5f')
# print(labels)

# for i, (name1, v1) in enumerate(vecs):
#     for k, (name2, v2) in enumerate(vecs):
#         sim = model.similarity(name1, name2)
#         print(name1, name2, sim)
#         array[i, k] = sim
#         # print(name1, name2, cal_cos_similarity(v1, v2))
#
# np.savetxt("array.csv", array, fmt='%.5f')
#
# from sklearn.cluster import KMeans
#
# kmeans_model = KMeans(n_clusters=2, random_state=10).fit(array)
# labels = kmeans_model.labels_
# np.savetxt("kmeans.csv", np.array(labels), fmt='%.5f')
# print(labels)
