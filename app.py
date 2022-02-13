from flask import Flask, request, jsonify
from flask_cors import CORS
import model

app = Flask(__name__)
CORS(app)

@app.route('/anime', methods = ['GET'])
def recommend_animes():
    res = model.results(request.args.get("anime_name"))
    return jsonify(res)


@app.route("/")
def hello():
    return "Hello from the machine learning model API"

if __name__ == '__main__':
    app.run(debug = True)