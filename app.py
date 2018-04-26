from flask import Flask, jsonify
from flask import request
from skimage import io
from detect_logo import *

app = Flask(__name__)

@app.route('/get_logo')
def get_logo():
    image = request.args.get('url')
    image = io.imread(image)
    return detect_logo(image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000)
