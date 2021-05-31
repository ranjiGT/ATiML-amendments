# `Advanced Topics in Machine Learning` ![](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)

## Data Science block :label:
- [`A1.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A1.1.rmd) - __Exploratory Data Analysis__
- [`A1.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A1.2.Rmd) - __Case Study on _20 newsgroup___
- [`A1.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A1.3.Rmd) - __Data Science Pipeline__
- [`A2.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A2.1.Rmd) - __Feature Selection (_Filter Techniques_)__
- [`A2.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A2.2.Rmd) - __Case Study on _Excess alcohol consumption among students___
- [`A2.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A2.3.Rmd) - __Feature Scaling__
- [`A2.4`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A2.4.Rmd) - __Feature Scaling on _k Nearest Neighbor___
- [`A4.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A4.1.Rmd) - __Data sampling techniques & strategies__
- [`A4.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A4.2.Rmd) - __Model selection and evaluation _(Grid Search & Cross-validation)___
- [`A4.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A4.3.Rmd) -  __Model comparison _(using Learning curves)___
- [`A4.4`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A4.4.Rmd) - __Statistical comparison of classifiers using _Dietterich's 5x2cv paired t-test___

## Semi-Supervised Learning :label:
- [`A5.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A5.1.Rmd) - __Linear Learning Machines__
- [`A5.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A5.2.Rmd) - __Dual Representation in LLM__
- [`A5.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A5.3.Rmd) - __Learning decision function using LLM__
- [`A5.4`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A5.4.Rmd) - __Support Vector Machines (SVM)__
- [`A6.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A6.1.Rmd) - __Semi-Supervised Learning__
- [`A6.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A6.2.Rmd) - __Propogating 1-NN__
- [`A6.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A6.3.Rmd) - __Self-Training__
- [`A6.4`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A6.4.Rmd) - __Generative Models__
- [`A7.1`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A7.1.Rmd) - __S3VM__
- [`A7.2`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A7.2.Rmd) - __Branch & Bound algorithm__
- [`A7.3`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A7.3.Rmd) - __Graph-based SSL__
- [`A7.4`](https://github.com/ranjiGT/ATiML-amendments/blob/main/ML2/A7.4.Rmd) - __Multiview Algorithms__

## Kernel Matrix
```python
def k(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            res[i][j] = np.dot(X[i], X[j]) ** 2
    return res
```

## Perceptron workflow

```python
def max_norm(X):
    norms = []
    for i in range(X.shape[0]):
        norms.append(np.linalg.norm(X[i]))
    return max(norms) ** 2
max_norm = max_norm(X)
b = 0 #given 
alpha = np.zeros(X.shape[0]) #given 
print('Itr#   ', '---alpha vec---', '     ---b')
for k in range(5):
    for i in range(X.shape[0]):
        s = 0
        for j in range(X.shape[0]):
            s += (alpha[j] * y[j] * res[j][i])
        s += b
        s *= y[i]
        if s <= 0:
            alpha[i] += 1
            b += (y[i] * (max_norm))        
        print(i+1, ' --- ', alpha, ' --- ', b)
    print()
print("alpha vector: ",alpha)
print("b value: ", b)
```
