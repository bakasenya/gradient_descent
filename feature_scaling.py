import numpy as np

from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_common import dlc
np.set_printoptions(precision=2)


# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

def z_score_normalization(X):
    
    mu = np.mean(X, axis=0)
    
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu) / sigma
    
    return (X_norm, mu, sigma)


X_norm, X_mu, X_sigma = z_score_normalization(X_train)

print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1}")