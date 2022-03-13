from face_store import FaceStore
import sys

def cluster(fmatch_iter):
    c = {}
    for l1, l2, dist in fmatch_iter:
        if l1 not in c:
            c[l1] = []
        c[l1].append(l2)
    print("got clusters: ", len(c.keys()))
    return c

def print_html():
    if len(sys.argv) < 4:
        raise "invalid args; 3 expected"
    db_f, html_f, max_d = sys.argv[1], sys.argv[2], float(sys.argv[3])
    fstor = FaceStore(db_f)
    fq = fstor.find_matches(dist_max=max_d)
    c = cluster(fq)
    f = open(html_f, "w")
    for k, v in c.items():
        f.write("<div class='cluster'>")
        f.write("<img src='%s'/>" % k)
        for i in v:
            f.write("<img src='%s'/>" % i)
        f.write("</div></br>")
    f.close()

print_html()
