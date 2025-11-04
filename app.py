from flask import Flask, render_template, Response, request
from asl_recognizer import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    user_name = request.args.get('name', 'Orang Hebat')
    lang = request.args.get('lang', 'id')
    return Response(generate_frames(user_name, lang),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
