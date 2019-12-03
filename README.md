# Holistic NLP Evaluator for Information Retrieval Systems

During the course of reviewing various NLP solutions for a particular use case, one thing I found I wish I had was a tool that would allow me to do a holistic apples to apples comparison of all the different Information Retrieval Systems. These types of problems, where you put some text into your model and get some other text out of it, are known as sequence to sequence or string transduction problems. Unfortunately for folks who are not familiar with NLP, there’s no straightforward answer about what metrics should be used to evaluate the model. Even for someone familiar with NLP, no single metric provides a holistic appraisal as every metric has major drawbacks, especially when applied to tasks that it was never intended to evaluate.
Bottomline: NLP can’t be measured as a whole, but rather specific to what one is trying to achieve.

## Goals

 1. Create an application that can provide a holistic and interpretable evaluation of an NLP model for the specific use case of information retrieval.
 2. The application will be retrieval method agnostic, i.e. it will be able to evaluate the NLP model irrespective of whether it’s a similarity-based model or a probabilistic model
 
## Specifications

Information Retrieval is an empirically defined problem. Therefor, this evaluator will use user feedback gleaned in two different ways to evaluate the NLP model:
 1. Directly from users, using a three star feedback mechanism defined as follows:
     * Three stars - exact match
     * Two stars - relevant match
     * One star - irrelevant match
 2. From linkage information between the documents created by the users to indicate relevance, i.e. whenever one document is referenced by another document it represents at least a two star relevance in the above scale.

The application will take the user feedback as input, generate an evaluation query set based on this feedback and when the results of an NLP model are provided as input to the application it will generate an analysis of the NLP model

# Instructions to run evalution 

## Setup

The following python libraries need to be installed for the evaluation to run:
 1. DateTime
 2. Numpy
 3. Scikit-Learn
 4. Pandas

```bash
# Ensure your pip is up to date
pip install --upgrade pip

# install DateTime!
pip install DateTime

# install Numpy!
pip install numpy

# install Scikit-Learn!
pip install -U scikit-learn

# install Pandas!
pip install pandas
```


## Step 1. 

Clone this github repository.

## Step 2.

To see a demo run of the evaluator on some demo data, simply execute the python program IR_EVALUATOR.py:

```
(base) ATBASU-M-45BA:CS410-Project-IR-EVALUATOR-master atbasu$ python3 IR_EVALUATOR.py
| Warning! Since insufficient model output files were provided, this evaluation will run on the demo dataset
| Evaluation in Progress
|------------------------------------------------------------------------
| + Loaded model output for evaluation
| + Evaluating IR...
| + Completed evaluation
|------------------------------------------------------------------------
| Report Card:
|------------------------------------------------------------------------
| * Average Normalized Decreased Cumulative Gain:  0.505
| * Cross_Entropy_Loss_Function Score(average):  0.723
| * Relevance Quality Indicator (percentage):  55.37
| * Relevance Quality Indicator (average):  0.009
| * Average Precision Score:  0.121
| * Average Recall Score:  0.009
| * Average F Score:  0.016
| * Precision Recall Area Under Curve: 0.10310705631336159
|------------------------------------------------------------------------
| Saving dataframes to file
| Ranked Relevance Quality Analysis : ./ranked_relevance_quality_analysis_2019-12-03.csv
| Set-based Relevance Quality Analysis : ./set-based_relevance_quality_analysis_2019-12-03.csv
|------------------------------------------------------------------------
(base) ATBASU-M-45BA:CS410-Project-IR-EVALUATOR-master atbasu$ 
```

The demo datasets are stored in the following folders:

1. Evaluation Datasets
2. Feedback Datasets
3. Model Output

To understand more about what is in these datasets and how they are used please read on.

## Step 3.

Extract the evluation dataset ranked_relevance_quality_indicator.json from the Evaluation Datasets folder [here](https://github.com/atbasu/CS410-Project-IR-EVALUATOR/tree/master/Evaluation%20Datasets)

If you convert the json file to a csv file, the dataset will look something like what you see below:

|           | query | result |
|-----------|-------|:------:|
|0|684523967|623517439|
|1|684523967|628883675|
|2|684523967|629261277|
|3|684523967|629498957|
|4|684523967|629828141|

## Step 4. 

For each query result pair from the downloaded dataset use the information retrieval system to get the relevance score. The resulting dataset should look something like what's shown below:

|           | query | result | score |
|-----------|-------|:------:|-------|
|0|684523967|623517439|0.20858895705521474|
|1|684523967|628883675|0.35582822085889576|
|2|684523967|629261277|0.6134969325153374|
|3|684523967|629498957|0.656441717791411|
|4|684523967|629828141|0.6319018404907976|

## Step 5. 

Store the resulting dataset as a json file with the following structure:
```
[
    {
        "query" : "684523967",
        "result" : "623517439",
        "score" : 0.20858895705521474
    },
    {
        "query" : "684523967",
        "result" : "628883675",
        "score" : 0.35582822085889576
    },
    {
        "query" : "684523967",
        "result" : "629261277",
        "score" : 0.6134969325153374
    },
    .
    .
    .
]
```

There are a couple of demo files in the "Model Output" folder.

## Step 6. 

Extract the evluation dataset set-based_relevance_quality_indicator.json from the Evaluation Datasets folder [here](https://github.com/atbasu/CS410-Project-IR-EVALUATOR/tree/master/Evaluation%20Datasets)

If you convert the json file to a csv file, the dataset will look something like what you see below:

| | query | 
|-|-------|
|0|632407033|
|1|633738153|
|2|634348243|
|3|634588015|
|4|634618119|

## Step 7. 

For each of these queries in the downloaded dataset return the top 10 documents identified by the IR system. The resulting dataset will look something like what's shown below:

| | query | results |
|-|-------|:------:|
|0|632407033|681625716;630939469;683043174;619202453;6...|
|1|633738153|680642087;638502841;682268779;639169265;6...|
|2|634348243|680184214;637376729;637412303;680435946;6...|
|3|634588015|681317797;637351667;637749307;637414593;6...|
|4|634618119|621044225;684209635;634135015;682045177;6...|

<b>Note:</b> Since the data will need to be uploaded as a csv, the list of top 10 documents should be seprated by a ';'

## Step 8. 

Store the resulting dataset as a json file with the following structure:
```
[
    {
        "query" : "632407033",
        "results" : ["681625716", "630939469", "683043174", "619202453", "6.."]
    },
    {
        "query" : "633738153",
        "results" : ["680642087", "638502841", "682268779", "639169265", "6.."]
    },
    {
        "query" : "634348243",
        "results" : ["680184214", "637376729", "637412303", "680435946", "6.."]
    },
    .
    .
    .
]
```

## Step 9.

Now execute the python program IR_EVALUATOR.py:

> (base) ATBASU-M-45BA:CS410-Project-IR-EVALUATOR-master atbasu$ python3 IR_EVALUATOR.py "path to json file generated in step 5" "path to json file generated in step 8"

## Step 10.

The result of the analysis will be generated on the terminal prompt and two csv files will be generated as shown below:
```
|------------------------------------------------------------------------
| Saving dataframes to file
| Ranked Relevance Quality Analysis : ./ranked_relevance_quality_analysis_2019-12-03.csv
| Set-based Relevance Quality Analysis : ./set-based_relevance_quality_analysis_2019-12-03.csv
|------------------------------------------------------------------------
```
To understand how these csv files can be converted into dataframes and used for further analysis of the performance of the IR system, run the python notebook "IR_EVALUATOR.py":

> (base) ATBASU-M-45BA:Project atbasu$ jupyter notebook

And refer to the "Results of Evaluation" section.
