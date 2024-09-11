from distutils.log import error
import numpy as np
from scipy.stats import norm
import torch, torch.nn
import numpy as np
from xgboost import XGBClassifier
import torch


class NaiveBayes(torch.nn.Module):
    def __init__(self, num_features, num_classes, std):
        super(NaiveBayes, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x):

        try:
            mask = x[:,self.num_features:]
            x = x[:,:self.num_features]
        except IndexError:
            error("Classifier expects masking information to be concatenated with each feature vector.")

        y_classes = list(range(self.num_classes))

        output_probs = torch.zeros((len(x), self.num_classes))

        for y_val in y_classes:

            ## PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1)
            p_x_y = torch.where((x >= 0) & (x < 1), torch.ones(x.shape), torch.zeros(x.shape))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:,y_val:y_val+3] = torch.transpose(
                torch.Tensor(np.array([norm.pdf(x[:,y_val], y_val % 2, self.std), 
                        norm.pdf(x[:,y_val+1], (y_val // 2) % 2, self.std), 
                        norm.pdf(x[:,y_val+2], (y_val // 4) % 2, self.std)])), 0, 1)

            # Compute joint probability over masked features
            p_xo_y = torch.prod(torch.where(torch.gt(mask, 0), p_x_y, torch.tensor(1).float()), dim=1)

            p_y = 1 / self.num_classes

            output_probs[:,y_val] = p_xo_y * p_y


        return torch.divide(output_probs, torch.squeeze(torch.dstack([torch.sum(output_probs, axis=1)]*self.num_classes)))
    
    def predict(self, x):
        return self.forward(x)

    
    


class classifier_xgb_dict():
    def __init__(self, output_dim, input_dim, subsample_ratio, X_train, y_train):
        """
        Input:
        output_dim: Dimension of the outcome y
        input_dim: Dimension of the input features (X)
        subsample_ratio: Fraction of training points for each boosting iteration

        Output:
        A dictionary of classifiers, predicting probabilities over y classes
        """
        self.xgb_model_dict = {}
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.subsample_ratio = subsample_ratio
        self.X_train_numpy = X_train.numpy()
        self.y_train_numpy = y_train.argmax(dim=1).numpy()
        
    def __call__(self, X, idx):
        n = X.shape[0]
        probs = torch.zeros((n, self.output_dim))
        for i in range(n):
            # Which mask?
            mask_i = X[i][self.input_dim:]
            nonzero_i = mask_i.nonzero().squeeze()
            mask_i_string = ''.join(map(str, mask_i.long().tolist()))
            # Is the mask in the dictionary?
            if mask_i_string not in self.xgb_model_dict:
                self.xgb_model_dict[mask_i_string] = XGBClassifier(n_estimators=250, max_depth=5, random_state=29, n_jobs=8)
                X_train_subset = self.X_train_numpy[:,nonzero_i].reshape(self.X_train_numpy.shape[0],-1)
                idx = np.random.choice(X_train_subset.shape[0], int(X_train_subset.shape[0]*self.subsample_ratio), replace=False)
                self.xgb_model_dict[mask_i_string].fit(X_train_subset[idx], self.y_train_numpy[idx])
            # Prediction
            probs[i] = torch.from_numpy(self.xgb_model_dict[mask_i_string].predict_proba(X[i, nonzero_i].numpy().reshape(1, -1)))
        return probs


class classifier_ground_truth():
    def __init__(self, num_features=20, num_classes=8, std=0.3):
        self.gt_classifier = NaiveBayes(num_features=num_features, num_classes=num_classes, std=std)
        
    def __call__(self, X, idx):
        return self.gt_classifier.predict(X)


class classifier_xgb():
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        
    def __call__(self, X, idx):
        return torch.tensor(self.xgb_model.predict_proba(X))


