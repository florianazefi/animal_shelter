from animal_shelter.features import add_features
from animal_shelter.data import load_data

animal_outcomes = load_data('../data/train.csv')
with_features = add_features(animal_outcomes)
with_features.head()
