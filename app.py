from flask import Flask, render_template, request
import detect as sd

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    result = sd.predict_spam(email_text)
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)