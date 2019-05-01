import sqlite3, hashlib
def register(username, password, permission):
    db = sqlite3.connect("logins.db")
    c = db.cursor()
    c.execute("INSERT INTO logins VALUES(?, ?, ?)", [username, hashlib.sha256(password.encode()).digest(), permission]);
    db.commit()
    db.close()
def delete(username):
    db = sqlite3.connect("logins.db")
    c = db.cursor()
    c.execute("DELETE FROM logins WHERE username=?", [username])
delete("doctor1")
