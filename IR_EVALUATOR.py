from datetime import datetime
from operator import itemgetter
import numpy as np
from sklearn import metrics
import pandas as pd
import sys
import json
import warnings

warnings.filterwarnings("ignore")

rr_df_csv = f"ranked_relevance_quality_analysis_{datetime.now().strftime('%Y-%m-%d')}.csv"
ir_df_csv = f"set-based_relevance_quality_analysis_{datetime.now().strftime('%Y-%m-%d')}.csv"
eval_rr_model_op = "./Model Output/ranked_relevance_quality_indicator.json"
eval_ir_model_op = "./Model Output/set-based_relevance_quality_indicator.json"
eval_rr_feedback = "./Feedback Datasets/ranked_relevance_quality_indicator.json"
eval_ir_feedback = "./Feedback Datasets/set-based_relevance_quality_indicator.json"

def getModelOutput():
    with open(eval_rr_model_op, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())
    rr_df = pd.DataFrame(data)

    with open(eval_ir_model_op, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    ir_df = pd.DataFrame(data)
    return(rr_df, ir_df)

def add_feedback(df1):
    with open(eval_rr_feedback, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    df2 = pd.DataFrame(data)
    df_final = pd.merge(df1, df2, how = 'inner', left_on=['query','result'], right_on=['query', 'result'])
    df_final = df_final.apply(lambda x: x.str.strip())
    df_final.score = df_final.score.apply(lambda x : float(x.replace("'","")))
    df_final.feedback = df_final.feedback.apply(lambda x : int(x.replace("'","")))
    return df_final

def retina_val(df):
    return df[df['query'].isin([doc for doc in df['query'].unique().tolist() if len(set(df[df['query'] == doc].feedback)) != 1])]

def num_of_star(list_star):
    star_1 = 0
    star_2 = 0
    star_3 = 0
    for x in list_star:
        if x == 1:
            star_1 +=1
        elif x ==2:
            star_2 += 1
        elif x == 3:
            star_3 +=1
    return star_1, star_2, star_3

def final_eval_dataset(df_f):
    list_retina = []
    df_final = retina_val(df_f)
    for doc in df_final['query'].unique().tolist():
        dic_ret = {}
        dic_ret['query'] = doc
        len_star = len(df_final[df_final['query'] == doc].feedback.tolist())
        feedback_list = [int(x) for x in df_final[df_final['query'] == doc].feedback.tolist()]
        score_list = [float(x) for x in df_final[df_final['query'] == doc].score.tolist()]
        new_list = []
        for x, y in zip(score_list, feedback_list):
            new_list.append([x,y])
        list_cs_sorted = sorted(new_list, key=itemgetter(0), reverse=True)
        list_fb_sorted = sorted(new_list, key=itemgetter(1), reverse=True)
        list_fb_rev_sorted = sorted(new_list, key=itemgetter(1), reverse=False)
        celf = 0
        dcg = 0
        idcg = 0
        nidcg = 0
        cs_sum = 0
        for index in range(len_star):
            if list_cs_sorted[index][1] != 1:
                celf += -np.log(list_cs_sorted[index][0])
            else:
                celf += -np.log(1-list_cs_sorted[index][0])
            try:
                if list_cs_sorted[index][1] != 1:
                    dcg += list_cs_sorted[index][1]/np.math.log(index+2, 2)
                else:
                    dcg += 0
            except:
                dcg += 0
            try:
                if list_fb_sorted[index][1] != 1:
                    idcg += list_fb_sorted[index][1]/np.math.log(index+2, 2)
                else:
                    idcg += 0
            except:
                idcg +=0
            try:
                if list_fb_rev_sorted[index][1] != 1:
                    nidcg += list_fb_rev_sorted[index][1]/np.math.log(index+2, 2)
                else:
                    nidcg += 0
            except:
                nidcg += 0
        try:
            ndcg = (dcg - nidcg)/(idcg-nidcg)
        except:
            ndcg = 'pass'
        feedbacks = num_of_star(df_final[df_final['query'] == doc].feedback.tolist())
        dic_ret['results'] = df_final[df_final['query'] == doc].result.tolist()
        dic_ret['scores'] = df_final[df_final['query'] == doc].score.tolist()
        dic_ret['feedback'] = df_final[df_final['query'] == doc].feedback.tolist()
        dic_ret['cross_entropy_loss'] = celf/len_star
        dic_ret['dcg'] = dcg
        dic_ret['ideal_dcg'] = idcg
        dic_ret['non_ideal_dcg'] = nidcg
        dic_ret['normalized_dcg'] = ndcg
        dic_ret['num_of_feedback'] = len_star
        dic_ret['3_star'] = feedbacks[2]
        dic_ret['2_star'] = feedbacks[1]
        dic_ret['1_star'] = feedbacks[0]
        list_retina.append(dic_ret)
    df = pd.DataFrame(list_retina)
    df.to_csv(rr_df_csv, index=False)
    return df

def csv_creation_and_analysis(df_input):
    df_int = add_feedback(df_input)
    df = final_eval_dataset(df_int)
    list_temp = []
    for x, y, z,u in zip(df.scores.tolist(), df.feedback.tolist(), df.normalized_dcg.tolist(), df.cross_entropy_loss.tolist()):
        avg_cos_sim1 = []
        avg_cos_sim23 = []
        for index in range(len(y)):
            if y[index] == 1:
                avg_cos_sim1.append(x[index])
            else:
                avg_cos_sim23.append(x[index])
        list_temp.append([np.mean(avg_cos_sim1),np.mean(avg_cos_sim23), float(z), float(u)])
    df_scatter = pd.DataFrame(list_temp, columns = ['average1','average23','normalized_dcg','cross_entropy_loss'])
    df_scatter = df_scatter[pd.notnull(df_scatter.average1) & pd.notnull(df_scatter.average23)]
    df_scatter['delta'] = df_scatter.average23 - df_scatter.average1
    df_scatter['validation'] = np.where(df_scatter['delta'] > 0, True, False)
    return df.normalized_dcg, df_int, df_scatter

#add code to generate Inferred Relevance Quality Indictor data
def fetch_inferred_pivot_dataset():
    with open(eval_ir_feedback, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    ir_mn = pd.DataFrame(data)
    return ir_mn

### Add the inferred relevance quality indicator code
def precision_recall_fpr(ir_df):
    """
    This function calculates the precision results of the each query document.
    """
    ir_mn = fetch_inferred_pivot_dataset()
    ir_df.drop_duplicates('query', keep = 'first', inplace=True)
    ir_mn.drop_duplicates('query', keep = 'first', inplace=True)
    dict_pre = {}
    dict_recall = {}
    dict_fscore = {}
    for x in ir_df['query'].unique().tolist():
        list_pre_total = []
        list_recall_total = []
        for y in ir_df[ir_df['query'] == x].results.tolist()[0]:
            count = 0
            list_pre = ir_df[ir_df['query'] == x].results.tolist()[0]
            list_recall = ir_mn[ir_mn['query'] == x].result.tolist()[0]
            for z in list_pre:
                if z in [l for l in list_recall if l != str(y)]:
                    count +=1
            list_pre_total.append(count/len(list_pre))
            list_recall_total.append(count/len(list_recall))

        precision = sum(list_pre_total)/len(list_pre_total)
        recall = sum(list_recall_total)/len(list_recall_total)
        dict_pre[x] = precision
        dict_recall[x] = recall
        try:
            dict_fscore[x] =  2*((precision * recall)/(precision + recall))
        except:
            dict_fscore[x] = 0
    ir_df['precision'] = ir_df['query'].map(dict_pre)
    ir_df['recall'] = ir_df['query'].map(dict_recall)
    ir_df['f_score'] = ir_df['query'].map(dict_fscore)
    return ir_df

def gen_report(rr_ndcg,rr_df_scatter,ir_df):
    #Ranked Relevance Quality Indicator
    print ('| * Average Normalized Decreased Cumulative Gain: ', round(np.mean(rr_ndcg), 3))

    #Semantic Relevance Quality Indicator
    print ('| * Cross_Entropy_Loss_Function Score(average): ', round(np.mean(rr_df_scatter.cross_entropy_loss), 3))
    print ('| * Relevance Quality Indicator (percentage): ',
           round(rr_df_scatter[rr_df_scatter.validation == True].__len__() / len(rr_df_scatter) * 100, 2))
    print ('| * Relevance Quality Indicator (average): ', round(np.mean(rr_df_scatter.delta), 3))

    #Inferred Relevance Quality Indicator
    print ('| * Average Precision Score: ', round(np.mean(ir_df.precision), 3))
    print ('| * Average Recall Score: ', round(np.mean(ir_df.recall), 3))
    print ('| * Average F Score: ', round(np.mean(ir_df.f_score), 3))
    area = metrics.auc(sorted(ir_df.recall, reverse=True), sorted(ir_df.precision, reverse=True))
    print('| * Precision Recall Area Under Curve:', area)

if __name__ == '__main__':

    if len(sys.argv) == 5:
        eval_rr_model_op = sys.argv[1]
        eval_rr_feedback = sys.argv[2]
        eval_ir_model_op = sys.argv[3]
        eval_ir_feedback = sys.argv[4]
    elif len(sys.argv) == 3:
        print(
            "| Notice! Since insufficient model output files were provided, this evaluation will use the demo feedback dataset")
        eval_rr_model_op = sys.argv[1]
        eval_ir_model_op = sys.argv[2]
    else:
        print("| Notice! Since insufficient model output files were provided, this evaluation will run on the demo dataset")

    print("| Evaluation in Progress")
    print("|------------------------------------------------------------------------")
    rr_df, ir_df = getModelOutput()
    print("| + Loaded model output for evaluation")
    print("| + Evaluating IR...")
    rr_ndcg, rr_df_int, rr_df_scatter = csv_creation_and_analysis(rr_df)
    ir_df = precision_recall_fpr(ir_df)
    ir_df.to_csv(ir_df_csv, index=False)
    print("| + Completed evaluation")
    print("|------------------------------------------------------------------------")
    print("| Report Card:")
    print("|------------------------------------------------------------------------")
    gen_report(rr_ndcg, rr_df_scatter, ir_df)
    print("|------------------------------------------------------------------------")
    print("| Saving dataframes to file\n| Ranked Relevance Quality Analysis : ./" + rr_df_csv + "\n| Set-based Relevance Quality Analysis : ./" + ir_df_csv + "\n|------------------------------------------------------------------------")