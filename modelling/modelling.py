from model.randomforest import RandomForest

def model_predict(data, df, name):
    # Create a RandomForest instance and run train, predict, evaluate
    model = RandomForest(name, data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.get_X_test())
    model_evaluate(model, data)

def model_evaluate(model, data):
    model.print_results(data)