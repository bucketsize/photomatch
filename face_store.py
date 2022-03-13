import sqlite3
from datetime import datetime as dt
from functools import reduce

class Box:
    def __init__(self, box):
        self.box = box

    def __repr__(self):
        bs = join(";", self.box) 
        return "Box[%s]" % bs


def join(sep, ls):
    s = ""
    for i in ls:
        if s == "":
            s = s+str(i)
        else:
            s = s+sep+str(i)
    return s        

def box_adapter(box):
    s = join(";", box.box) 
    return s.encode('ascii')

def box_converter(s):
    box = list(map(int, s.split(b";")))
    return box

sqlite3.register_adapter(Box, box_adapter)
sqlite3.register_converter("box", box_converter)

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
        cur.execute('''
            CREATE TABLE if not exists faces (
                        date text,
                        image_path text,
                        face_path text,
                        confidence real,
                        face_box box
            )
        ''')
            
        cur.execute('''
            CREATE TABLE if not exists matches (
                        face_path_1 text,
                        face_path_2 text,
                        confidence real
            )
        ''')
        con.commit()
        self.store = (con, cur)

    def face_row(self, face):
        box = Box(face["box"])
        now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
        face_id = face["face_path"]
        return (now, face["image_path"], face_id, face["confidence"], box)

    def match_row(self, match):
        return match

    def save_faces(self, faces):
        con, cur = self.store
        f_list = list(map(self.face_row, faces))
        cur.executemany("insert into faces values (?, ?, ?, ?, ?)", f_list)
        con.commit()
        return 0

    def save_matches(self, matches):
        con, cur = self.store
        m_list = list(map(self.match_row, matches))
        cur.executemany("insert into matches values (?, ?, ?)", m_list)
        con.commit()
        return 0

    def find_faces(self):
        con, cur = self.store
        cur.execute('select * from faces')
        return FaceIter(cur)

    def find_image_path(self, image_path):
        con, cur = self.store
        cur.execute("select date from faces where image_path = '%s'" % image_path)
        result = cur.fetchall()
        return result

def test():
    store = FaceStore()
    status = store.save_faces("/var/foo/121.jpg", [[42,22,33,15], [56,40,32,21]])
    print(status)

    res = store.find_faces()
    for face in res:
        print(face)

    res = store.find_image_path("/var/foo/121.jpg")
    for im in res:
        print(im)

# test()
