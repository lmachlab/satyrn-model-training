# satyrn-model-training

make_dataset.py takes the folder of .csv and .xlsx files and extracts the "claims" and "# of claims " columns from each file to compile them into one .csv file with these two columns. It should be noted that the column name used for this was "# of claims " rather than "# of claims", so this may need to be changed in future use.

clean_dataset.py removes rows from the created dataset that don't have both claims and # of claims values.

remove_data.py removes rows containing specific thresholded values in the # of claims column.

bert_train.py processes the dataset and uses it to finetune BERT. The BERT model used was bert-base-uncased (https://huggingface.co/bert-base-uncased) and I used this article as a reference https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
