import traceback

class RunModel():
    t1 = '\t'
    t2 = '\t\t'
    t3 = '\t\t\t'
    def __init__(self, model, model_params):
        self.model_name = model.__name__
        self.model_params = model_params
        self.model = self.build_model(model, model_params)

    def build_model(self, model, model_params):
        print("="*50)
        print(f"Building model {self.model_name}")
        
        try:
            model = model(**model_params)
        except Exception as e:
            err = f"Exception caught while building model for {self.model_name}:"
            catch_and_throw(e, err)
        return model
    
    def fit(self, *args, **kwargs):
        print(f"Training {self.model_name}...")
        print(f"{self.t1}Using hyperparameters: ")
        [print(f"{self.t2}{n} = {get_name(v)}")for n, v in self.model_params.items()]
        try: 
            return self._fit(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while training model for {self.model_name}:"
            catch_and_throw(e, err)
            
    def _fit(self, X, y, metrics=None, pass_y=False):
        if pass_y:
            self.model.fit(X, y)
        else:
             self.model.fit(X)
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix='Train')
        return scores
    
    def evaluate(self, *args, **kwargs):
        print(f"Evaluating {self.model_name}...")
        try:
            return self._evaluate(*args, **kwargs)
        except Exception as e:
            err = f"Exception caught while evaluating model for {self.model_name}:"
            catch_and_throw(e, err)
        

    def _evaluate(self, X, y, metrics, prefix=''):
        preds = self.model.predict(X)
        scores = self.get_metrics(y, preds, metrics, prefix)      
        return scores
    
    def predict(self, X):
        try:
            preds = self.model.predict(X)
        except Exception as e:
            err = f"Exception caught while making predictions for model {self.model_name}:"
            catch_and_throw(e, err)
            
        return preds
    
    def get_metrics(self, y, y_hat, metrics, prefix=''):
        scores = {}
        for name, metric in metrics.items():
            score = metric(y, y_hat)
            display_score = round(score, 3)
            scores[name] = score
            print(f"{self.t2}{prefix} {name}: {display_score}")
        return scores
    
def get_name(obj):
    try:
        if hasattr(obj, '__name__'):
            return obj.__name__
        else:
            return obj
    except Exception as e:
        return obj
    
def catch_and_throw(e, err):
    trace = traceback.format_exc()
    print(err + f"\n{trace}")
    raise e