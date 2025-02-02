import pickle

model_path = r"C:\Users\gudeg\OneDrive\Documents\Random_forest_model.pkl"

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Now you can use the model (for example, print it)
print(model)
