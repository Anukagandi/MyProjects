from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form for user input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        input_text = request.form['text']
        
        # Vectorize the input text
        input_vector = vectorizer.transform([input_text])
        
        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        
        # Create response
        response = {
            "input": input_text,
            "sentiment": prediction
        }
        
        return render_template('result.html', input=input_text, sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)
