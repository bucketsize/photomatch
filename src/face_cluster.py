from face_store import FaceStore
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch 
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle

def load_data(db_f):
    fstor = FaceStore(db_f)
    fq = fstor.find_faces()
    idx, dat, fps = [], [], []
    for face_id,_,face_path,_,_,em in fq:
        print(">> ", face_id, em.shape)
        idx.append(face_id)
        fps.append(face_path)
        dat.append(em[None,:])
    a_dat = torch.cat(dat, 0)
    return (idx, fps, a_dat)

def distmean(e1, e2):
    "params: Tensor: torch.Size([512])"
    # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # d = cos(e1, e2).item()
    # d = LA.norm(e1-e2).item()
    d = (e1 - e2).norm().item()
    return d

def cluster_distmean(data, dist=0.5):
    s = []
    for i, ei in enumerate(data):
        for j, ej in enumerate(data):
            d = distmean(ei, ej)
            if 0 < d and d < dist:
                s.append((i, j, d))
    s.sort(key=lambda x:x[2])
    c, r = {}, []
    for i in range(0, len(data)):
        c[i] = [i]
    for i, j, d in s:
        if (j not in r):
            c[i].append(j)
            r.append(j)
    print(c)
    print(len(c), len(r))
    
    index = {}
    for i, js in enumerate(c.values()):
        for j in js:
            index[j] = i

    print(len(index.values()))
    return ("distmean-"+str(dist) , len(c), index.values())

def cluster_using(tag, model, data):
    model.fit(data)
    clsts = model.predict(data)
    return (tag, len(clsts), clsts)

def adj_list(labels, fps):
    c = {}
    for l, p in list(zip(labels, fps)):
        if l not in c:
            c[l] = []
        c[l].append(p)
    return c
        
def print_html(html_f, tag, c):
    f = open(html_f+"_"+tag+".html", "w")
    for k, v in c.items():
        f.write("<div class='cluster'>")
        f.write("<div class='label'><span>%s</span><br/></div>" % k)
        for i in v:
            f.write("<img src='%s'/>" % i)
        f.write("</div></br></br>")
    f.close()

def main():
    if len(sys.argv) < 4:
        raise "invalid args; 3 expected"
    db_f, html_f, max_d, n_clusters = sys.argv[1],sys.argv[2],float(sys.argv[3]), int(sys.argv[4])
    (_, L, X) = load_data(db_f)
    for tag, n, ls in [
            cluster_using("AffPr", AffinityPropagation(damping=0.57), X),
            cluster_using("BIRCH", Birch(threshold=0.01, n_clusters=n_clusters), X),
            cluster_using("KMeans", KMeans(n_clusters=n_clusters), X),
            cluster_using("GaussM",GaussianMixture(n_components=n_clusters), X),
    ]:
        print(tag, n, ls)
        c = adj_list(ls, L)
        print_html(html_f, tag, c)

main()
    # -- plot --
    # plt.close("all")
    # plt.figure(1)
    # plt.clf()
    # colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    # for k, col in zip(range(n_clusters_), colors):
    #     class_members = labels == k
    #     cluster_center = X[cluster_centers_indices[k]]
    #     plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
    #     plt.plot(
    #         cluster_center[0],
    #         cluster_center[1],
    #         "o",
    #         markerfacecolor=col,
    #         markeredgecolor="k",
    #         markersize=14,
    #     )
    #     for x in X[class_members]:
    #         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    # plt.title("Estimated number of clusters: %d" % n_clusters_)
    # plt.show()
