"""Predeclared task-nested model selection. Never select on tasks 6 or 7."""
import json
import platform
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from src.config import ROOT,FEATURES
from src.data import validate_and_prepare
from src.provenance import sha256_bytes
from src.phase4_ml import CONFIGS,make_pipeline,score_metrics,recall_threshold,threshold_curve

OUT=ROOT/'reports/phase4'
FAMILIES=list(dict.fromkeys(c['family'] for c in CONFIGS))


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def tasks(frame):return sorted(int(t) for t in frame.task_id.unique())


def oof(frame,config,audit):
    scores=np.full(len(frame),np.nan)
    for fit_idx,valid_idx in GroupKFold(3).split(frame,groups=frame.task_id):
        fit,valid=frame.iloc[fit_idx],frame.iloc[valid_idx]
        assert not set(tasks(fit)) & set(tasks(valid))
        model=make_pipeline(config).fit(fit[config['features']],fit.label)
        scores[valid_idx]=model.predict_proba(valid[config['features']])[:,1]
        audit.append(dict(config=config['id'],fit_tasks=tasks(fit),validation_tasks=tasks(valid)))
    assert np.isfinite(scores).all()
    return scores


def trial(frame,config,scores):
    threshold=recall_threshold(frame.label,scores,.7)
    measures=score_metrics(frame.label,scores,threshold)
    worst=min(score_metrics(g.label,scores[frame.task_id.to_numpy()==task],threshold)['recall']
              for task,g in frame.groupby('task_id'))
    return dict(config=config['id'],family=config['family'],threshold=threshold,
                worst_task_recall=worst,**measures)


def best(trials,family):
    return min((t for t in trials if t['family']==family),key=lambda t:(
        t['false_alarms_per_1000_normal'],-t['worst_task_recall'],-t['f1']))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/'data/cad_tcm.csv',low_memory=False)
    raw['_row']=np.arange(1,len(raw)+1)
    # Keep conflict/duplicate handling independent of previously seen evaluation labels.
    dev=validate_and_prepare(raw[raw.task_id<6]).reset_index(drop=True)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        sklearn=sklearn.__version__,data_sha256=sha256_bytes((ROOT/'data/cad_tcm.csv').read_bytes()),
        configs=CONFIGS,outer_test_groups=[[0,3],[1,4],[2,5]],inner='GroupKFold(3)',
        selection='pooled validation Recall>=0.70, minimum false alarms; worst recall then F1 tie break',
        calibration='sigmoid fit to inner OOF scores/labels only within outer training; evaluate on untouched outer test',
        adoption_rule='All task recalls >=0.70; pooled FAR<baseline; pooled F1>=baseline; worst task F1>=baseline. No independent external validation.',
        folds=[])
    dump('experiment_manifest.json',manifest)
    predictions=[];selection=[];calibration=[]
    def collect(model,protocol,fold,frame,scores,warning,caution,config):
        predictions.append(pd.DataFrame(dict(model=model,protocol=protocol,fold=fold,
            task_id=frame.task_id.to_numpy(),source_row_id=frame._row.to_numpy(),label=frame.label.to_numpy(),
            score=scores,threshold=warning,caution_threshold=caution,config=config)))
    for fold,test_tasks in enumerate(manifest['outer_test_groups']):
        train=dev[~dev.task_id.isin(test_tasks)].reset_index(drop=True)
        test=dev[dev.task_id.isin(test_tasks)].reset_index(drop=True)
        fold_audit=dict(fold=fold,fit_tasks=tasks(train),test_tasks=tasks(test),inner=[])
        trials=[];scores_by_config={}
        for config in CONFIGS:
            scores=oof(train,config,fold_audit['inner']);scores_by_config[config['id']]=scores
            result=trial(train,config,scores);trials.append(result)
            selection.append(dict(stage='inner_selection',fold=fold,**result))
            print('inner',fold,config['id'],round(result['false_alarms_per_1000_normal'],2),flush=True)
        for family in FAMILIES:
            selected=best(trials,family);config=next(c for c in CONFIGS if c['id']==selected['config'])
            scores=scores_by_config[config['id']]
            warning=selected['threshold'];caution=recall_threshold(train.label,scores,.9)
            model=make_pipeline(config).fit(train[config['features']],train.label)
            test_scores=model.predict_proba(test[config['features']])[:,1]
            collect(family,'nested_selected',fold,test,test_scores,warning,caution,config['id'])
            collect(family,'nested_fixed_0.5',fold,test,test_scores,.5,min(caution,.5),config['id'])
            # Sigmoid mapping sees only outer-training inner OOF scores and labels.
            calibrator=LogisticRegression(C=1.,max_iter=1000).fit(scores.reshape(-1,1),train.label)
            calibrated=calibrator.predict_proba(test_scores.reshape(-1,1))[:,1]
            calibrated_train=calibrator.predict_proba(scores.reshape(-1,1))[:,1]
            ct=recall_threshold(train.label,calibrated_train,.7)
            collect(family,'nested_sigmoid',fold,test,calibrated,ct,
                    recall_threshold(train.label,calibrated_train,.9),config['id'])
            calibration.append(dict(model=family,fold=fold,calibration='raw',**score_metrics(test.label,test_scores,warning)))
            calibration.append(dict(model=family,fold=fold,calibration='sigmoid',**score_metrics(test.label,calibrated,ct)))
        # Frozen baseline RF configurations, refit on each outer training set.
        for name,threshold in [('rf51',.5),('rf36',.839762745020482)]:
            config=next(c for c in CONFIGS if c['id']==name)
            model=make_pipeline(config).fit(train[config['features']],train.label)
            ts=model.predict_proba(test[config['features']])[:,1]
            collect(name,'nested_baseline',fold,test,ts,threshold,
                    min(threshold,recall_threshold(train.label,scores_by_config[name],.9)),name)
        manifest['folds'].append(fold_audit);dump('experiment_manifest.json',manifest)
        pd.concat(predictions,ignore_index=True).to_csv(OUT/'predictions_partial.csv',index=False)
    # Final selection uses only development task OOF, committed before accessing 6/7.
    final_trials=[];final_scores={};final_audit=[]
    for config in CONFIGS:
        scores=oof(dev,config,final_audit);final_scores[config['id']]=scores
        result=trial(dev,config,scores);final_trials.append(result)
        selection.append(dict(stage='final_development_selection',fold=-1,**result))
        print('final',config['id'],flush=True)
    decisions={family:best(final_trials,family) for family in FAMILIES}
    manifest['final_inner']=final_audit;manifest['final_decisions']=decisions
    manifest['selection_frozen_utc']=datetime.now(timezone.utc).isoformat()
    dump('experiment_manifest.json',manifest)
    pd.DataFrame(selection).to_csv(OUT/'selection_trials.csv',index=False)
    # Only now run previously-seen tasks 6/7, never for selection or calibration.
    previous=validate_and_prepare(raw[raw.task_id>=6]).reset_index(drop=True)
    registry={};policies={};curves=[]
    q=dev.loc[dev.label.eq(0),FEATURES].quantile([.05,.25,.5,.75,.95]).T
    q.columns=['p05','q25','median','q75','p95']
    for family,decision in decisions.items():
        config=next(c for c in CONFIGS if c['id']==decision['config'])
        scores=final_scores[config['id']];threshold=decision['threshold']
        caution=recall_threshold(dev.label,scores,.9)
        model=make_pipeline(config).fit(dev[config['features']],dev.label)
        policy=dict(caution_threshold=caution,warning_threshold=threshold,
                    basis='Development task OOF score cutoffs targeting pooled recall 0.90 / 0.70; research only')
        policy['version']='sha256:'+sha256_bytes(json.dumps(policy,sort_keys=True).encode())
        report=score_metrics(dev.label,scores,threshold)
        report.update(accuracy=float(((scores>=threshold)==dev.label).mean()),
            confusion_matrix=[[report['tn'],report['fp']],[report['fn'],report['tp']]],
            threshold_used_for_metrics=threshold,training_date_utc=datetime.now(timezone.utc).isoformat(),
            n_features=len(config['features']),train_count=len(dev),test_count=len(dev),
            split_method='Development task OOF selection metrics (not independent validation); see phase4 nested results',
            feature_importance={},source_kind='cad_tcm',notices=['연구용 AI 점수이며 실설비 고장 확률이나 안전 기준이 아닙니다.'])
        estimator=model.named_steps['estimator']
        if hasattr(estimator,'feature_importances_'):
            importance=estimator.feature_importances_
        elif hasattr(estimator,'coef_'):
            importance=np.abs(estimator.coef_[0])
        else:
            importance=None
        if importance is not None:
            report['feature_importance']=pd.Series(importance,index=config['features']).sort_values(ascending=False).to_dict()
        bundle=dict(pipeline=model,features=config['features'],normal_stats=q.to_dict('index'),
                    report=report,source_kind='cad_tcm',risk_policy=policy,model_family=family)
        path=ROOT/f'models/phase4_{family}.joblib';joblib.dump(bundle,path,compress=3)
        registry[family]=dict(file=path.name,sha256=sha256_bytes(path.read_bytes()),config=config['id'],threshold=threshold,risk_policy=policy)
        collect(family,'development_oof_selection',-1,dev,scores,threshold,caution,config['id'])
        collect(family,'reused_6_7_research',-1,previous,model.predict_proba(previous[config['features']])[:,1],threshold,caution,config['id'])
        curve=threshold_curve(dev.label,scores);curve['model']=family;curves.append(curve)
        for target in [.7,.8,.9]:
            cut=recall_threshold(dev.label,scores,target)
            selection.append(dict(stage='equal_recall',fold=-1,family=family,target_recall=target,
                                  threshold=cut,**score_metrics(dev.label,scores,cut)))
    # Bind empirical risk cutoffs to the exact old artifacts without rewriting them.
    for name,path in [('rf51','cad_tcm_rf.joblib'),('rf36','cad_tcm_candidate_rf.joblib')]:
        scores=final_scores[name]
        policy=dict(caution_threshold=recall_threshold(dev.label,scores,.9),
                    warning_threshold=recall_threshold(dev.label,scores,.7),
                    basis='Development task OOF recall targets 0.90 / 0.70; research only')
        policy['version']='sha256:'+sha256_bytes(json.dumps(policy,sort_keys=True).encode())
        policies['sha256:'+sha256_bytes((ROOT/'models'/path).read_bytes())]=policy
        bundle=joblib.load(ROOT/'models'/path)
        collect(name,'development_oof_baseline',-1,dev,scores,bundle['report']['threshold_used_for_metrics'],
                min(bundle['report']['threshold_used_for_metrics'],policy['caution_threshold']),name)
        collect(name,'reused_6_7_stored',-1,previous,bundle['pipeline'].predict_proba(previous[bundle['features']])[:,1],
                bundle['report']['threshold_used_for_metrics'],min(bundle['report']['threshold_used_for_metrics'],policy['caution_threshold']),name)
        # Actual risk-stage counts use independent warning cutoff, not binary threshold.
        collect(name,'development_oof_risk_policy',-1,dev,scores,policy['warning_threshold'],policy['caution_threshold'],name)
    dump('model_registry.json',dict(candidates=registry,legacy_risk_policies=policies,default_model='cad_tcm_rf.joblib',adopted=False))
    pd.DataFrame(selection).to_csv(OUT/'selection_trials.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration_comparison.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/'threshold_curves.csv.gz',index=False,compression='gzip')
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    task_rows=[];summary=[];risk=[]
    for (name,protocol),group in pred.groupby(['model','protocol']):
        task_stats=[]
        for task,g in group.groupby('task_id'):
            values=score_metrics(g.label,g.score,g.threshold)
            task_rows.append(dict(model=name,protocol=protocol,task_id=int(task),**values));task_stats.append(values)
            levels=np.where(g.score>=g.threshold,'경고',np.where(g.score>=g.caution_threshold,'주의','정상'))
            for level in ['정상','주의','경고']:
                labels=g.label.to_numpy()[levels==level]
                risk.append(dict(model=name,protocol=protocol,task_id=int(task),risk_level=level,
                                 normal=int((labels==0).sum()),anomaly=int((labels==1).sum())))
        summary.append(dict(model=name,protocol=protocol,**score_metrics(group.label,group.score,group.threshold),
            worst_task_f1=min(t['f1'] for t in task_stats),worst_task_recall=min(t['recall'] for t in task_stats),
            worst_task_far=max(t['false_alarms_per_1000_normal'] for t in task_stats),
            macro_task_ap=float(np.mean([t['average_precision'] for t in task_stats])),
            task_f1_std=float(np.std([t['f1'] for t in task_stats]))))
    pd.DataFrame(task_rows).to_csv(OUT/'task_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(risk).to_csv(OUT/'risk_counts.csv',index=False,encoding='utf-8-sig')
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat();dump('experiment_manifest.json',manifest)
    (OUT/'predictions_partial.csv').unlink(missing_ok=True)
    print(pd.DataFrame(summary).query("protocol in ['nested_selected','nested_baseline']").to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
