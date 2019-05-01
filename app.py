import os, sys, hashlib
from extra import register, delete
from bottle import route, run, request, app, response, hook
import sqlite3, prediction
db = sqlite3.connect("logins.db")
c = db.cursor()
doctor = 1
lab = 2
options = {
  "certfile": 'cacert.pem',
  "keyfile": 'privkey.pem'
}
@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
class CorsPlugin():
    name = "cors"
    api = 2
    def apply(self, fn, context):
        def _cors(*args, **kwargs):
            response.set_header('Access-Control-Allow-Origin', '*')
            response.add_header('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
            response.headers["Access-Control-Allow-Origin"] = "*";
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, OPTIONS";
            response.headers["Access-Control-Allow-Headers"] = "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token"
            if request.method != "OPTIONS":
                return fn(*args, **kwargs)
        return _cors

def sha256(string):
    return hashlib.sha256(string.encode()).digest()
def login(username, password):
    uenc = sha256(username)
    penc = sha256(password)
@route('/version')
def index():
    return sys.version
@route('/upload', method="POST")
def index():
    upload = request.files.get("files")
    path = "uploads/" + hashlib.sha256(os.urandom(16)).hexdigest()[:16] + ".png";
    upload.save(path);
    print("yes")
    return prediction.run();
@route("/login", method="POST")
def index():
    username = request.forms.get('username')
    password = request.forms.get('password')
    if(not username or not password):
        return 'invalid'
    r = c.execute("SELECT * FROM 'logins' WHERE username=? AND passhash=?", [username, sha256(password)])
    r = list(r)
    if(len(r) > 0): #valid
        permission = r[0][2]
        return 'valid' + str(permission)
    elif(len(list(r)) == 0):
        return 'incorrect'
    else:
        return 'error'
app = app()
app.install(CorsPlugin())
port = os.environ.get('PORT', 5000)
run(host='0.0.0.0', port=port)
