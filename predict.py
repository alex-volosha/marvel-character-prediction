import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('characters_sex')

@app.route('/predict', methods=['POST'])
def predict():
    character = request.get_json()

    X = dv.transform([character])
    y_pred = model.predict_proba(X)[0,1]
    characters_sex = y_pred >= 0.4

    result = {
        'Probability is': float(y_pred),
        'Female Character': bool(characters_sex)
    }

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 9696))
    app.run(host="0.0.0.0", port=port)