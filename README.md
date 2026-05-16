# part-1-neural-network-analysis
This project is on building and analysing a supervised feed forward neural network (multi layer perceptron) to predict customer 'churn' (target variable).  

## Dataset Exploration  
The dataset shows significant class imbalance, with 98.45% representing the retained customers and only 1.55% representing churned customers. Because this class is so small, a standard network can guess "No Churn" for every customer. In this project, the model evaluates the **Confusion Matrix**, **Precision** and **Recall** metrics, to evaluate whether the network is truly learning the features and not cheating with the imbalanced class ratio.

## Data Pre-processing  
The dataset contains **2000 rows** (customer records) and **17 columns** (features). The attributes are grouped into **Categorical Variables** and **Numerical Variables**.  
The data is pre-processed by,  
- dropping the `customer_id` from the training matrix since unique alphanumeric identifiers have no predictive power.
- passing the text columns through a One-Hot Encoder. `drop='first'` is used to simply network calculation and by-pass the dummy variable.
- using `StandardScaler` is used to make uniform the numerical feature variances.
- splitting the dataset in an 80:20 train:test , `stratify=y` is used to distribute the churned examples evenly.

## Model Architecture  
The model architecture passes batch inputs from 24 neuron network to 12 neuron hidden layer using **ReLU**, to handle non linear combinations. **Binary Crossentropy** function evaluates the output probablity against the true state.  
For backpropagation and parameter updates, the **Adam** optimiser reads gradients and adjusts the model parameters over the epochs automatically, to decrease the loss score.


## Results

- **Neural Network Accuracy:** 98.25%
- **Neural Network Precision:** 0.0
- **Neural Network Recall:** 0.0
- **Neural Network F1 Score:** 0.0

* **Training Loss:** 0.0314 | **Training Accuracy:** 98.87%
* **Testing Loss:** 0.0809 | **Testing Accuracy:** 98.25%

**Confusion Matrix**
```text
               Predicted Not Churned    Predicted Churned
Actual Not Churned       393                    1
Actual Churned             6                    0
```
