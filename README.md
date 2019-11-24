# CS410-Project-IR-EVALUATOR
NLP Evaluation framework with sample data

# Instructions to run evalution 

## Step 1. 

Download the evluation dataset for Ranked Relevance Quality Indicator test [here](https://scripts.cisco.com:443/ui/use/NLP_EVAL?autorun=true&action=Download&evalType=ranked_relevance_quality_indicator&dev=true&devAdditionalTasks=%5B%5D)
The dataset will look something like what you see below:

|           | query | result |
|-----------|-------|:------:|
|0|684523967|623517439|
|1|684523967|628883675|
|2|684523967|629261277|
|3|684523967|629498957|
|4|684523967|629828141|

## Step 2. 

For each query result pair from the downloaded dataset get the relevance score generated the NLP model. The resulting dataset will look something like what's shown below:

|           | query | result | score |
|-----------|-------|:------:|-------|
|0|684523967|623517439|0.20858895705521474|
|1|684523967|628883675|0.35582822085889576|
|2|684523967|629261277|0.6134969325153374|
|3|684523967|629498957|0.656441717791411|
|4|684523967|629828141|0.6319018404907976|

## Step 3. 

Upload the resulting dataset [here](https://scripts.cisco.com:443/ui/use/NLP_EVAL?action=Upload&evalType=ranked_relevance_quality_indicator&dev=true&devAdditionalTasks=%5B%5D). Fill in the options that have not already been pre-filled and then hit the "run" button.

You have to upload it as a CSV file consisting of three collumns and one header row - query,result,score. Once uploaded it will be converted into a list of dictionaries as shown below and stored in a mongodb. For reference you can download a sample csv file [here](https://cisco.box.com/s/ijrlfmk0hgccuyovtqgwmgjksg9bt4mu)

## Step 4.

Once the file is uploaded the tool with output the name of the collection where your data has been stored, copy that name and update the variable "eval_rr_model_op" below and then run the code cell to confirm the data is present:

## Step 5. 

Download the evluation dataset for Inferred Relevance Quality Indicator test [here](https://scripts.cisco.com:443/ui/use/NLP_EVAL?autorun=true&action=Download&evalType=inferred_relevance_quality_indicator&dev=true&devAdditionalTasks=%5B%5D)
The dataset will look something like what you see below:

| | query | 
|-|-------|
|0|632407033|
|1|633738153|
|2|634348243|
|3|634588015|
|4|634618119|

## Step 6. 

For each of these queries in the downloaded dataset return the top 10 documents identified by the NLP model. The resulting dataset will look something like what's shown below:

| | query | results |
|-|-------|:------:|
|0|632407033|681625716;630939469;683043174;619202453;6...|
|1|633738153|680642087;638502841;682268779;639169265;6...|
|2|634348243|680184214;637376729;637412303;680435946;6...|
|3|634588015|681317797;637351667;637749307;637414593;6...|
|4|634618119|621044225;684209635;634135015;682045177;6...|

<b>Note:</b> Since the data will need to be uploaded as a csv, the list of top 10 documents should be seprated by a ';'

## Step 7. 

Upload the resulting dataset [here](https://scripts.cisco.com:443/ui/use/NLP_EVAL?action=Upload&evalType=inferred_relevance_quality_indicator&dev=true&devAdditionalTasks=%5B%5D). Fill in the options that have not already been pre-filled and then hit the "run" button.

You have to upload it as a CSV file consisting of two collumns and one header row - query,results. Once uploaded it will be converted into a list of dictionaries as shown below and stored in a mongodb. For reference you can download a sample csv file [here](https://cisco.box.com/s/ieq31rxrg0uvuvagj82d7mr4j0o1flny)

## Step 8.

Once the file is uploaded the tool with output the name of the collection where your data has been stored, copy that name and update the variable "eval_ir_model_op" below and then run the code cell to confirm the data is present:

## Step 9.

Now run the following code cell to complete the evaluation:


