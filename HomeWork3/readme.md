The purpose of this assignment is to design a model that accepts a text passage
along with a question as inputs and returns the correct answer found within the
provided passage.
GitHub Repository: For dataset & code, visit the GitHub repository here:
https://github.com/vishnusai-nara/DeepLearning/tree/main/HomeWork3
Dataset:
This project employs the Spoken Question Answering Dataset (SQuAD), which
includes 37,111 records. Each record comprises a passage used as context and
a set of questions related to it, each with corresponding answers. One passage
often has multiple associated questions..
Data Preprocessing:
In the preprocessing stage, the text undergoes tokenization—a process that
segments the text into smaller, manageable units or tokens, making it compatible
for model processing. I applied the DistilBertTokenizerFast from the
distilbert-base-uncased pre-trained model to perform tokenization on contexts,
questions, and answers. For each answer, the starting and ending tokens are
identified, a vital step for accurate training. Additionally, the text is normalized to
maintain consistency in answer formatting, which aids in more dependable
evaluation.
Model:
For this task, the distilbert-base-uncased model from Hugging Face is utilized.
This model is a lightweight, optimized version of BERT built on a transformer
structure, trained on vast amounts of English text data through a self-supervised
approach focusing on Masked Language Modeling (MLM). With 66 million
parameters, DistilBERT strikes a balance between computational efficiency and
robust performance.
Training:
During training, I monitored and documented the model's loss and accuracy by
comparing its predictions to the true positions. After completing 6 epochs with a
learning rate of 2e-5, using the Adam optimizer and Focal Loss to handle class
imbalance, the final step involved calculating the average loss and accuracy
across all batches. The trained model and tokenizer were then saved for future
use.
Testing:
The model's accuracy on the test data is measured through F1 and Word Error
Rate (WER) metrics. For testing, you have the option to either train the model by
executing the relevant scripts or to download the pretrained model files from the
link provided below. Afterward, you can set the paths to your training and testing
datasets within the code to carry out the evaluation.
Google drive link of generated files:
https://drive.google.com/drive/folders/101So1y_pPwg5gJYOX4zoz-s6MawqpJ19
?dmr=1&ec=wgc-drive-hero-goto
Results:
Different models with some improvements, base model(distilbert-base-uncased),
medium model(doc stride,scheduler) and strong model(Other Pretrained Model)
are used and tested on three different data sets
Initially the base model, distilbert-base-uncased, was trained and tested on three
datasets with three different noise conditions i.e, three different Word Error
Rates which are No Noise -22.73%, Noise V1 - 44.22%, Noise V2 - 54.82%. To
enhance its performance, several improvements were applied:
Doc Stride: The doc_stride parameter was set to 128 to create overlapping
segments within long documents. This overlap prevents answers from being split
across chunks, allowing the model to more accurately identify correct responses.
Learning Rate Decay scheduler: An ExponentialLR scheduler was used to
gradually decrease the learning rate with a decay rate of 2e-2. This approach
enables finer adjustments to the model’s parameters as it approaches the optimal
solution, improving overall convergence. The base model was tested using both
the doc stride setting and the learning rate scheduler.
Additionally, another pretrained model, deepset/bert-base-cased-squad 2,
was evaluated to compare its performance with base - model.
