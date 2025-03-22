# Iris-Flower-Classification
This repository contains a notebook that demonstrates the process of classifying Iris flowers using a Support Vector Classifier (SVC) from the scikit-learn library. The dataset used in this project is the famous Iris flower dataset, which is a well-known dataset in the field of machine learning and statistics.

The Iris flower dataset, also known as Fisher's Iris dataset, consists of 150 samples from three species of Iris flowers: Iris setosa, Iris virginica, and Iris versicolor. For each flower sample, four features were measured:
Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)

Each species has 50 samples, making a total of 150 samples in the dataset.
Here is a visual representation of the three different species of Iris flowers:

Iris Setosa

![image](https://github.com/user-attachments/assets/573da810-3752-4cfe-80a9-5040b0014633)

Iris Versicolor

Iris Virginica
![image](https://github.com/user-attachments/assets/d68aa307-8982-4ea2-8b5b-16725ad2c1da)

Get the Data
The Iris dataset is loaded using the seaborn library:
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
iris.species.value_counts()

Results
After tuning the model, we observed that the performance remained similar to the original model, as expected given the small size of the dataset. However, this provides a good foundation for improving the model with larger and more complex datasets.

Requirements:

To run the notebook, you will need the following libraries:
seaborn
matplotlib
pandas
scikit-learn

You can install them using pip:

pip install seaborn matplotlib pandas scikit-learn

License
This project is licensed under the MIT License - see the LICENSE file for details.

