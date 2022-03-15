import io
import sys
import torch
from torch import tensor
import sqlite3
from datetime import datetime as dt
from functools import reduce

def pt_tensor_adapter(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()

def pt_tensor_converter(bs):
    buf = io.BytesIO(bs)
    return torch.load(buf) 

sqlite3.register_adapter(torch.Tensor, pt_tensor_adapter)
sqlite3.register_converter("pt_tensor", pt_tensor_converter)

class FaceIter:
    def __init__(self, cursor):
        self.count = 1
        self.cursor = cursor

    def __iter__(self):
        return self

    def __next__(self):
        r = self.cursor.fetchone()
        if r is None:
            raise StopIteration
        self.count += 1
        return r 

class FaceStore:
    def __init__(self, db_file=":memory:"):
        con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.executescript('''
            CREATE TABLE if not exists faces (
                        date text,
                        image_path text,
                        face_path text,
                        confidence real,
                        face_box pt_tensor 
            );
            CREATE TABLE if not exists embeddings (
                        face_path text,
                        embedding pt_tensor
            );
            CREATE TABLE if not exists matches (
                        face_path_1 text,
                        face_path_2 text,
                        confidence real
            );
        ''')
        con.commit()
        self.con = con
        self.cur = cur

    def face_row(self, face):
        """
        input (image_path, face_path, box, confidence, tensor)
        """
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        row = (now, face[0], face[1], float(face[3]), tensor(face[2]))
        print(">> ", row)
        return row
    
    def embedding_row(self, fe):
        return (fe[0][1], fe[1])

    def match_row(self, match):
        return match

    def save_faces(self, faces, embeddings):
        f_list = list(map(self.face_row, faces))
        self.cur.executemany("insert into faces values (?, ?, ?, ?, ?)", f_list)
        e_list = list(map(self.embedding_row,
                          list(zip(faces, embeddings))))
        self.cur.executemany("insert into embeddings values (?, ?)", e_list)
        self.con.commit()
        return 0

    def save_matches(self, matches):
        m_list = list(map(self.match_row, matches))
        self.cur.executemany("insert into matches values (?, ?, ?)", m_list)
        self.con.commit()
        return 0

    def find_faces(self):
        self.cur.execute('select * from faces')
        return FaceIter(self.cur)
    
    def find_embeddings(self):
        self.cur.execute('select * from embeddings')
        return FaceIter(self.cur)

    def find_image_path(self, image_path):
        self.cur.execute("select date from faces where image_path = '%s'" % image_path)
        result = self.cur.fetchall()
        return result

    def find_matches(self, dist_max=0.3):
        self.cur.execute("select * from matches where confidence < %f" % dist_max)
        return FaceIter(self.cur)


def test():
    store = FaceStore(db_file="/var/tmp/1.db")
    status = store.save_faces(
        [("/var/foo/121.jpg", "/var/foo/f/1", tensor([42,22,33,15]), 0.98, []),
         ("/var/foo/121.jpg", "/var/foo/f/2", tensor([96,44,36,26]), 0.97, [])],
        [tensor([.2,.2,.3,.5]),
         tensor([.6,.0,.2,.1])]
    )
    print(status)

    res = store.find_faces()
    for face in res:
        print(face)

    res = store.find_image_path("/var/foo/121.jpg")
    for im in res:
        print(im)

def test_file():
    store = FaceStore(sys.argv[1])
    res = store.find_faces()
    for face in res:
        print(face)

#test()
test_file()
