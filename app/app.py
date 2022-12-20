from io import StringIO
import pandas as pd
import joblib

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

import pandas as pd
import joblib


app = FastAPI()


@app.get("/api/v1/ping")
def ping(n_times: int):
    return "pong" * n_times


@app.post("/api/v1/predict/")
def predict(file: UploadFile = File(...)):
    input_data = pd.read_csv(file.file)
    # TODO: Fill this in.

    # Process data. 
    X_test = input_data.rename({'AnimalType': 'animal_type', 
                                'SexuponOutcome': 'sex_upon_outcome'}, 
                               axis=1)
    simple_cols = ['animal_type', 'sex_upon_outcome']
    X_pred_dummies = pd.get_dummies(X_test.loc[:, simple_cols])

    # Load model.
    outcome_model = joblib.load('output/outcome_model.pickle')

    # Create predictions. 
    y_pred = outcome_model.predict_proba(X_pred_dummies)

    # Combine predictions with class names and animal name.
    classes = outcome_model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes)
    predictions = input_data[['Name']].join(proba_df)
    
    #predictions = input_data
    response = _convert_df_to_response(predictions)
    return response



def _convert_df_to_response(df: pd.DataFrame) -> StreamingResponse:
    """Convert a DataFrame to CSV response."""
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    return response
