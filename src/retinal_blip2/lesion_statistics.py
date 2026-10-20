"""Paired-original-group bootstrap of all binary lesion confusion metrics."""
from collections import defaultdict

import numpy as np

LESIONS = ['EX', 'HE', 'MA', 'SE']


def divide(numerator,denominator):
    return np.divide(numerator,denominator,out=np.full(np.shape(numerator),np.nan,dtype=float),where=denominator>0)


def calculate(counts):
    # Order: true-no/pred-no, true-no/pred-yes, true-no/invalid,
    # true-yes/pred-no, true-yes/pred-yes, true-yes/invalid.
    tn,fp,invalid_no,fn,tp,invalid_yes=np.moveaxis(counts,-1,0)
    positive_support=fn+tp+invalid_yes;negative_support=tn+fp+invalid_no
    metrics={
        'positive_precision':divide(tp,tp+fp),
        'positive_recall':divide(tp,positive_support),
        'positive_F1':divide(2*tp,2*tp+fp+fn+invalid_yes),
        'negative_precision':divide(tn,tn+fn),
        'negative_recall':divide(tn,negative_support),
        'negative_F1':divide(2*tn,2*tn+fp+invalid_no+fn),
        'accuracy':divide(tp+tn,counts.sum(axis=-1))}
    metrics['balanced_accuracy']=(metrics['positive_recall']+metrics['negative_recall'])/2
    dual=(metrics['positive_F1']+metrics['negative_F1'])/2
    metrics['binary_macro_F1']=np.where((positive_support>0)&(negative_support>0),dual,np.nan)
    return metrics


def lesion_intervals(rows,field='prediction',n_bootstrap=2000,seed=20261020):
    grouped=defaultdict(lambda:np.zeros((4,6),dtype=np.int64))
    for row in rows:
        for index,key in enumerate(LESIONS):
            truth=int(bool(row['evidence_truth'][key]));prediction=row['lesions'][key].get(field)
            column=(0 if prediction=='no' else 1 if prediction=='yes' else 2)+3*truth
            grouped[row['group_id']][index,column]+=1
    groups=sorted(grouped)
    counts=np.array([grouped[g] for g in groups],dtype=np.int64)
    if not len(groups):return {'status':'NOT_ESTIMABLE','reason':'no original-image groups'}
    total=counts.sum(axis=0);point=calculate(total)
    support={
        'positive_precision':((counts[:,:,1]+counts[:,:,4])>0).sum(axis=0),
        'positive_recall':(counts[:,:,3:6].sum(axis=-1)>0).sum(axis=0),
        'positive_F1':((counts[:,:,1]+counts[:,:,3]+counts[:,:,4]+counts[:,:,5])>0).sum(axis=0),
        'negative_precision':((counts[:,:,0]+counts[:,:,3])>0).sum(axis=0),
        'negative_recall':(counts[:,:,:3].sum(axis=-1)>0).sum(axis=0),
        'negative_F1':(counts[:,:,:4].sum(axis=-1)>0).sum(axis=0),
        'accuracy':np.full(4,len(groups),dtype=int)}
    support['balanced_accuracy']=np.minimum(support['positive_recall'],support['negative_recall'])
    support['binary_macro_F1']=support['balanced_accuracy']
    samples=np.random.RandomState(seed).randint(0,len(groups),size=(n_bootstrap,len(groups)))
    simulated=calculate(counts[samples].sum(axis=1))
    def interval(value,resamples,effective_groups):
        valid=np.isfinite(resamples);valid_n=int(valid.sum());invalid_n=int((~valid).sum())
        enough=len(groups)>=20 and effective_groups>=20 and valid_n>=max(100,int(.8*n_bootstrap)) and np.isfinite(value)
        return {'point':float(value) if np.isfinite(value) else None,
                'ci95':[float(x) for x in np.percentile(resamples[valid],[2.5,97.5])] if enough else None,
                'status':'ESTIMATED' if enough else 'NOT_ESTIMABLE_RELIABLY_SPARSE_SUPPORT' if effective_groups<20 else 'NOT_ESTIMABLE_RELIABLY',
                'effective_original_groups':int(effective_groups),'minimum_original_groups':20,
                'valid_resamples':valid_n,'invalid_resamples':invalid_n}
    per_class={}
    for index,key in enumerate(LESIONS):
        tn,fp,ino,fn,tp,iyes=map(int,total[index])
        per_class[key]={'images':tn+fp+ino+fn+tp+iyes,'positive_n':fn+tp+iyes,'negative_n':tn+fp+ino,
                        'predicted_positive_n':tp+fp,'predicted_negative_n':tn+fn,'invalid_prediction_n':ino+iyes,
                        'confusion':[tn,fp,ino,fn,tp,iyes],
                        'metrics':{metric:interval(values[index],simulated[metric][:,index],support[metric][index]) for metric,values in point.items()}}
    macro={metric:interval(np.mean(values),np.mean(simulated[metric],axis=1),int(support[metric].min())) for metric,values in point.items()}
    return {'status':'GROUPED_ORIGINAL_IMAGE_BOOTSTRAP','original_groups':len(groups),
            'n_bootstrap':n_bootstrap,'seed':seed,'per_lesion':per_class,'mean_over_four_lesions':macro,
            'interval_scope':'95% pointwise percentile; missing denominators remain undefined; all four lesions travel with original image',
            'confusion_order':['TN','FP','invalid_true_negative','FN','TP','invalid_true_positive']}
