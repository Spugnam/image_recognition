#!/usr/local/bin//python3

"""
Inspired from this blog:
https://medium.com/product-engineering-tophatter/classifying-marketplace-inventory-at-scale-with-machine-learning-af22930ee2c9
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from stop_words import get_stop_words
from sklearn.naive_bayes import MultinomialNB


data_folder = "../data"
categories_file = "categories.csv"
categories_file_path = os.path.join(data_folder, categories_file)

product_file = "cat_and_title.csv"
product_file_path = os.path.join(data_folder, product_file)


def main():

    ###########
    # Load data
    ###########

    categories = pd.read_csv(categories_file_path)
    print("Loaded {} categories".format(categories.shape[0]))

    products = pd.read_csv(product_file_path)
    print("Loaded {} products".format(products.shape[0]))

    # Fix column names
    remove_words = ['Category', 'Name -']
    for word in remove_words:
        categories.columns = categories.columns.str.replace(word, '')
    categories.columns = categories.columns.str.strip()

    products.columns = ['category_ID', 'title']

    #####################
    # Basic visualization
    #####################

    # Show sample
    categories.sample(2)
    products.sample(5)
    # Show top categories
    products['category_name'].value_counts()[:10]

    # Only use parent categories
    parent_categories = categories.loc[pd.isna(categories['Child 1']),
                                       ['ID', 'Name', 'Parent']]
    parent_categories.head()
    parent_IDs_dict = pd.Series(data=parent_categories['ID'].values,
                                index=parent_categories['Parent']).to_dict()
    parent_IDs_dict
# {'Animals': '55906718531b3b2b478b456c',
#  'Art & Collectibles': '559064a8531b3b95438b456c',
# ...
#  'Tickets': '559066c7531b3bab628b4568'}

    categories['Parent_ID'] = categories['Parent'].map(parent_IDs_dict)
    categories.head(1)
    parent_IDs_mapping_dict =\
        pd.Series(data=categories['Parent_ID'].values,
                  index=categories['ID']).to_dict(into=dict)
    parent_IDs_mapping_dict
# {'55906545531b3baa628b4568': '55906545531b3baa628b4568',
#  '559068f8531b3b093e8b4568': '55906545531b3baa628b4568',
#  '55906905531b3b93438b456e': '55906545531b3baa628b4568'}

    products['parent_category_ID'] = products['category_ID'].map(parent_IDs_mapping_dict)
    products.head(5)

    # Add category names to products df
    category_name_dict = pd.Series(data=categories['Name'].values,
                                   index=categories['ID']).to_dict()

    products['category_name'] = products['category_ID'].map(category_name_dict)
    products['parent_category_name'] =\
        products['parent_category_ID'].map(category_name_dict)
    # Reorder columns
    cols = products.columns
    products = products[[cols[1], cols[0], cols[2], cols[3], cols[4]]]
    products.head(n=3)
    # Save to file
    # product_categories.to_csv('../data/product_categories.csv')

    products['parent_category_name'].value_counts()

    # training/ test data
    X_train, X_val, y_train, y_val =\
        train_test_split(products['title'],
                         products['parent_category_name'],
                         test_size=0.2, random_state=10)

    # PolynomialNB model
    stop_words = get_stop_words('english') + get_stop_words('portuguese')
    cv = CountVectorizer(stop_words=stop_words,
                         lowercase=True,
                         max_df=0.5, min_df=2,
                         ngram_range=(1, 2))
    cv.fit(X_train, y=y_train)
    print("Vocabulary from training: {}".format(len(cv.vocabulary_)))

    counts = cv.transform(X_train)
    # print(counts[0])
    nb = MultinomialNB()
    nb.fit(counts, y_train)

    predictions = nb.predict(cv.transform(X_val))
    # print(predictions.head(n=10))

    correct = sum(predictions == y_val)
    incorrect = len(predictions) - correct
    print("{}/{} correct [{:.2%}]".format(correct,
                                          correct + incorrect,
                                          correct / (correct + incorrect)))

    # Inspect wrong predictions
    predictions_df = pd.DataFrame(data=list(zip(X_val, predictions, y_val)),
                                  columns=['title', 'Prediction', 'Label'])
    predictions_df.loc[predictions_df['Prediction'] != predictions_df['Label']]

    # LogisticRegression model
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    p = Pipeline(steps=[
        ('counts', CountVectorizer(stop_words=stop_words,
                                   lowercase=True,
                                   max_df=0.5, min_df=2,
                                   ngram_range=(1, 2))),
        ('lr', LogisticRegression())
        ])
    p.fit(X_train, y_train)
    predictions = p.predict(X_val)

    correct = sum(predictions == y_val)
    incorrect = len(predictions) - correct
    print("{}/{} correct [{:.2%}]".format(correct,
                                          correct + incorrect,
                                          correct / (correct + incorrect)))


if __name__ == "__main__":
    main()
