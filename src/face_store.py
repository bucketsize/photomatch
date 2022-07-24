import os
import io
import sys
import time
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
        print("using db", db_file)
        con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.executescript('''
            CREATE TABLE if not exists images (
                        image_id integer,
                        created text,
                        image_dir text,
                        image_name text,
                        image_hash text
            );
            CREATE TABLE if not exists faces (
                        face_id integer,
                        image_id integer,
                        face_path text,
                        confidence real,
                        face_box pt_tensor, 
                        embedding pt_tensor
            );
        ''')
        con.commit()
        self.con = con
        self.cur = cur
        self.index = int(time.time())

    def next_idx(self):
        self.index += 1
        return self.index

    def split_path(self, image_path):
        bna = os.path.basename(image_path)
        idr = image_path[0:(len(image_path)-len(bna))]
        return (idr, bna)

    def save_faces(self, faces):
        image_path,fs,es = faces
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        idr, bna = self.split_path(image_path)
        image_id = self.next_idx()
        image = (image_id, now, idr, bna, "hash")
        self.cur.execute("insert into images values (?, ?, ?, ?, ?)", image)
        for fi, ti in list(zip(fs, es)):
            face_id = self.next_idx()
            face = (face_id, image_id, fi[0], float(fi[2]), tensor(fi[1]), ti)
            # print(face)
            self.cur.execute("insert into faces values (?, ?, ?, ?, ?, ?)", face)
        self.con.commit()
        return 0

    def find_faces(self):
        self.cur.execute('select * from faces')
        return FaceIter(self.cur)
    
    def find_image_path(self, image_path):
        self.cur.execute("select created from images where image_dir = '%s' and image_name = '%s'"
                        % self.split_path(image_path))
        result = self.cur.fetchall()
        return result

def test():
    store = FaceStore()
    status = store.save_faces(
        (
            "/var/foo/121.jpg",
            [("/var/foo/f/1", tensor([42,22,33,15]), 0.98, []),
             ("/var/foo/f/2", tensor([96,44,36,26]), 0.97, [])],
            [tensor([.2,.2,.3,.5]),
             tensor([.6,.0,.2,.1])]
        )
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
#test_file()
