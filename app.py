from flask import Flask, render_template, request
import joblib

# Load models
bow_obj = joblib.load('./models/bag_of_words.lb')
model = joblib.load('./models/bernouliNB.lb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def data():
    if request.method == 'POST':
        message = request.form['info']  # get message from form
        email_vector = bow_obj.transform([message]).toarray()  # correct transformation

        prediction = model.predict(email_vector)[0]  # predict

        # Convert prediction to readable label
        label_dict = {'0': "Ham", '1': "Spam"}
        return render_template('output.html', output=label_dict[str(prediction)])

if __name__ == '__main__':
    app.run(debug=True)
