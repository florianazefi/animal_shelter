import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
 
import pandas as pd
import typer
import joblib

from animal_shelter.data import load_data

app = typer.Typer()


@app.callback()
def main() -> None:
    """Determine animal shelter outcomes."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def predict(input_path:Path, model_path: Path, output_path: Path) -> None:
    """Predicts the model performance for a given dataset."""
    
    typer.echo(f"Loading {input_path}")
    
    logger = logging.getLogger(__name__)
    logger.info("Loading input dataset from %s", input_path)
    
    test_dataset = load_data(input_path)
    logger.info("Found %i rows", len(test_dataset))
    
    X_test = test_dataset 
    
    outcome_model = joblib.load(model_path)
    
    simple_cols = ['animal_type', 'sex_upon_outcome']
    
    X_pred_dummies = pd.get_dummies(X_test.loc[:, simple_cols])
    y_pred = outcome_model.predict_proba(X_pred_dummies)
    
    classes = outcome_model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes)
    
    proba_df['id'] = test_dataset['id']
    reordered = proba_df[['id'] + classes]
    
    reordered.to_csv(output_path, index=False)
    
    logger.info(f"Wrote model to {output_path}")
    
   
    
@app.command()
def train(input_path: Path, model_path: Path) -> None:
    """Trains a model on the given dataset."""

    typer.echo(f"Loading {input_path}")

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset from %s", input_path)
    train_dataset = load_data(input_path)
    logger.info("Found %i rows", len(train_dataset))

    # TODO: Fill in your solution.
    # - Separate feature matrix X from target y
    X = train_dataset.drop('outcome_type', axis=1)
    y = train_dataset['outcome_type']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    
    # - add/remove features such that the feature matrix X is suitable for ML
    simple_cols = ['animal_type', 'sex_upon_outcome']

    X_train_dummies = pd.get_dummies(X_train.loc[:, simple_cols])
    
    # - Fit a model
    param_grid = {'C': [1E-3, 1E-2, 1E-1]}

    grid_search = GridSearchCV(LogisticRegression(), 
                               param_grid=param_grid, 
                               scoring='neg_log_loss')

    grid_search.fit(X_train_dummies, y_train)
    
    # - Log the final score
    best_model = grid_search.best_estimator_
    
    # - Save model
    joblib.dump(best_model, model_path)

    logger.info(f"Wrote model to {model_path}")
