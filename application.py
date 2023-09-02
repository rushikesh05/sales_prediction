from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        TV = float(request.form['TV'])
        Radio = float(request.form['Radio'])
        Newspaper = float(request.form['Newspaper'])

        # Make a prediction using the trained model
        prediction = model.predict([[TV, Radio, Newspaper]])

        return render_template('index.html', prediction=f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
