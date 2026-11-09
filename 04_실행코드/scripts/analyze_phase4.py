"""Task statistics and exportable figures; association is not causation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from src.config import ROOT,FEATURES
from src.phase4_ml import score_metrics

OUT=ROOT/'reports/phase4'


def run():
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'data/cad_tcm.csv',low_memory=False)
    counts=frame.groupby(['task_id','label']).size().unstack(fill_value=0).rename(columns={0:'normal',1:'anomaly'})
    counts['anomaly_ratio']=counts.anomaly/counts[['normal','anomaly']].sum(axis=1)
    counts.to_csv(OUT/'task_distribution.csv')
    stats=[];shifts=[]
    for (task,label),group in frame.groupby(['task_id','label']):
        reference=frame[(frame.task_id<6)&(frame.task_id!=task)&(frame.label==label)]
        for feature in FEATURES:
            vals=group[feature].dropna();ref=reference[feature].dropna()
            q=vals.quantile([.05,.25,.5,.75,.95])
            stats.append(dict(task_id=task,label=label,feature=feature,n=len(vals),mean=vals.mean(),std=vals.std(),
                              p05=q.loc[.05],q25=q.loc[.25],median=q.loc[.5],q75=q.loc[.75],p95=q.loc[.95]))
            iqr=ref.quantile(.75)-ref.quantile(.25)
            shifts.append(dict(task_id=task,label=label,feature=feature,reference='other development tasks 0-5, same label',
                median_shift_dev_iqr=(vals.median()-ref.median())/iqr if iqr else np.nan,
                ks_statistic=ks_2samp(ref,vals).statistic))
    pd.DataFrame(stats).to_csv(OUT/'feature_statistics.csv',index=False)
    shift=pd.DataFrame(shifts);shift.to_csv(OUT/'distribution_shift.csv',index=False)
    shift[shift.task_id.isin([3,4,5])].assign(abs_shift=lambda g:g.median_shift_dev_iqr.abs()).sort_values(
        ['task_id','label','abs_shift'],ascending=[True,True,False]).groupby(['task_id','label']).head(10).to_csv(OUT/'tasks_3_4_5_shift_details.csv',index=False)
    plt.style.use('dark_background')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.facecolor':'#071625',
                         'axes.facecolor':'#12263B','savefig.facecolor':'#071625','font.size':10})
    def save(fig,name):
        fig.tight_layout();fig.savefig(plots/name,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ratios=counts[['normal','anomaly']].div(counts[['normal','anomaly']].sum(axis=1),axis=0)
    ratios.plot.bar(stacked=True,ax=ax,color=['#23c4bd','#ffbc66'])
    ax.set(title='태스크별 실제 정상·이상 비율',xlabel='태스크',ylabel='태스크 내 비율')
    save(fig,'task_label_ratios.png')
    # Representative stand 3 variables; full 51-variable quantiles are exported.
    features=['force_3','torque_3','tension_3','roll_speed_3']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for feature,ax in zip(features,axes.flat):
        data=[frame.loc[(frame.task_id==t)&(frame.label==0),feature].to_numpy() for t in range(8)]
        ax.boxplot(data,showfliers=False,tick_labels=list(range(8)),patch_artist=True,
                   boxprops={'facecolor':'#23c4bd','alpha':.65},medianprops={'color':'#ffbc66'})
        unit={'force_3':'N','torque_3':'N·m','tension_3':'N','roll_speed_3':'원자료 수치 · 단위 미확인'}[feature]
        ax.set(title=feature+' · 정상 사례',xlabel='태스크',ylabel=unit)
    save(fig,'feature_distributions.png')
    if not (OUT/'predictions.csv.gz').exists():
        print('Task distributions done; run again after experiment for model figures');return
    predictions=pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    table=pd.read_csv(OUT/'task_metrics.csv')
    selected=table[table.protocol.eq('nested_selected')]
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for metric,ax in zip(['f1','recall'],axes):
        for model,group in selected.groupby('model'):
            ax.plot(group.task_id,group[metric],marker='o',label=model)
        ax.set(title=f'개발 태스크별 {metric} · 바깥 홀드아웃',xlabel='태스크',ylabel=metric,ylim=(0,1.02));ax.legend(fontsize=7)
    save(fig,'task_f1_recall.png')
    fig,ax=plt.subplots(figsize=(11,4))
    for model,group in selected.groupby('model'):
        ax.plot(group.task_id,group.false_alarms_per_1000_normal,marker='o',label=model)
    ax.set(title='태스크별 정상 1,000건당 오경보 · 바깥 홀드아웃',xlabel='태스크',ylabel='정상 1,000건당 오경보');ax.legend(fontsize=8)
    save(fig,'task_false_alarms.png')
    nested=predictions[predictions.protocol.eq('nested_selected')]
    points=[]
    for model,group in nested.groupby('model'):
        for point,column in [('warning_recall070','threshold'),('caution_recall090','caution_threshold')]:
            task_values=[score_metrics(g.label,g.score,g[column]) for _,g in group.groupby('task_id')]
            points.append(dict(model=model,operating_point=point,
                **score_metrics(group.label,group.score,group[column]),
                worst_task_recall=min(v['recall'] for v in task_values),
                worst_task_f1=min(v['f1'] for v in task_values)))
    pd.DataFrame(points).to_csv(OUT/'operating_point_comparison.csv',index=False,encoding='utf-8-sig')
    score_summary=nested.groupby(['model','task_id','label']).score.describe(percentiles=[.05,.25,.5,.75,.95])
    score_summary.to_csv(OUT/'score_distribution.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for (model,group),ax in zip(nested.groupby('model'),axes.flat):
        for label,color,offset in [(0,'#23c4bd',-.15),(1,'#ffbc66',.15)]:
            data=[group.loc[(group.task_id==t)&(group.label==label),'score'].to_numpy() for t in range(6)]
            bp=ax.boxplot(data,positions=np.arange(6)+offset,widths=.25,showfliers=False,patch_artist=True,
                boxprops={'facecolor':color,'alpha':.65},medianprops={'color':'white'})
        ax.set(title=model+' · 청록=정상 / 주황=이상',xlabel='태스크',ylabel='모델 점수',xticks=range(6),xticklabels=range(6))
    save(fig,'task_score_distributions.png')
    curves=pd.read_csv(OUT/'threshold_curves.csv.gz')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for model,group in curves.groupby('model'):
        sampled=group.iloc[::max(1,len(group)//1000)]
        axes[0,0].plot(sampled.recall,sampled.precision,label=model)
        axes[0,1].plot(sampled.threshold,sampled.recall,label=model)
        axes[1,0].plot(sampled.threshold,sampled.f1,label=model)
        axes[1,1].plot(sampled.threshold,sampled.false_alarms_per_1000_normal,label=model)
    for ax,title,x,y in zip(axes.flat,['Precision-Recall','Recall-Threshold','F1-Threshold','오경보-Threshold'],
                            ['Recall','Threshold','Threshold','Threshold'],['Precision','Recall','F1','정상 1,000건당 오경보']):
        ax.set(title=title+' · 개발 OOF 선택용',xlabel=x,ylabel=y);ax.legend(fontsize=7)
    save(fig,'threshold_analysis.png')
    calibration=pd.read_csv(OUT/'calibration_comparison.csv')
    calibration.groupby(['model','calibration'])[['brier','log_loss']].mean().to_csv(OUT/'calibration_summary.csv')
    # Same score bins, outer-held task labels only, no calibration fitting here.
    bins=np.linspace(0,1,11);rows=[]
    for (model,protocol),group in predictions[predictions.protocol.isin(['nested_selected','nested_sigmoid'])].groupby(['model','protocol']):
        codes=np.minimum(np.searchsorted(bins,group.score,side='right')-1,9)
        for b,g in group.groupby(codes):
            rows.append(dict(model=model,protocol=protocol,bin=int(b),n=len(g),mean_score=g.score.mean(),observed_anomaly_rate=g.label.mean()))
    pd.DataFrame(rows).to_csv(OUT/'calibration_bins.csv',index=False)
    print('Task and model analysis figures complete')


if __name__=='__main__':run()
