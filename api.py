from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from the machine learning model API"

if __name__ == '__main__':
    app.run(debug = True)