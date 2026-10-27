from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RESULTS=HERE/"results"; FIGURES=HERE/"figures"; FIGURES.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/"clarkson2_attention_model")); from train_attention import AttentionEncoder

TARGET="163748"; STAGE="stage_2"

def run_with_patch(model, x, layer, replacement=None):
    captured=[]
    def hook(_m,_i,out):
        if replacement is None: captured.append(out.detach())
        else:
            if out.ndim==3: return replacement.unsqueeze(0).expand(out.shape[0],-1,-1).to(out)
            return replacement.unsqueeze(0).expand(out.shape[0],-1).to(out)
    h=layer.register_forward_hook(hook)
    with torch.no_grad():
        z=model(x)
    h.remove()
    return z.detach().numpy(), captured

def collect_reference(model, x, layer):
    chunks=[]
    def hook(_m,_i,out): chunks.append(out.detach())
    h=layer.register_forward_hook(hook)
    with torch.no_grad(): model(x)
    h.remove()
    return torch.cat(chunks,dim=0).mean(dim=0)

def main():
    ck=torch.load(RESULTS/"attention_model_best.pt",map_location="cpu",weights_only=False)
    model=AttentionEncoder(); model.classifier=torch.nn.Linear(32,len(ck["users"])); model.load_state_dict(ck["model"]); model.eval()
    mean=np.array(ck["mean"],dtype=np.float32); std=np.array(ck["std"],dtype=np.float32)
    base=np.load(RESULTS/"sequences.npz"); bm=pd.read_csv(RESULTS/"sequence_metadata.csv",dtype={"user_id":str})
    later=np.load(RESULTS/"remaining_stage_sequences.npz"); lm=pd.read_csv(RESULTS/"remaining_stage_metadata.csv",dtype={"user_id":str})
    train_ix=np.where((bm.split=="train").to_numpy()&(bm.user_id==TARGET).to_numpy())[0]
    target_ix=np.where((lm.stage==STAGE).to_numpy()&(lm.user_id==TARGET).to_numpy())[0]
    if len(train_ix)==0 or len(target_ix)==0: raise RuntimeError(f"Missing target windows: train={len(train_ix)}, later={len(target_ix)}")
    Xtrain=torch.from_numpy(((base["X"][train_ix]-mean)/std).astype(np.float32)); Xtarget=torch.from_numpy(((later["X"][target_ix]-mean)/std).astype(np.float32))
    with torch.no_grad(): prototype=model(Xtrain).mean(0).numpy()
    baseline,_=run_with_patch(model,Xtarget,model.norm,None); baseline_dist=np.linalg.norm(baseline-prototype,axis=1)
    layers={"input_projection":model.proj,"transformer_layer_1":model.encoder.layers[0],"transformer_layer_2":model.encoder.layers[1],"post_encoder_norm":model.norm,"embedding_projection":model.embedding}
    ref={name:collect_reference(model,Xtrain,layer) for name,layer in layers.items()}
    # Wrong-user control uses another available user's early windows.
    control_user=next(u for u in bm.user_id.astype(str).unique() if u!=TARGET)
    control_ix=np.where((bm.split=="train").to_numpy()&(bm.user_id.astype(str).to_numpy()==control_user))[0][:len(train_ix)]
    Xcontrol=torch.from_numpy(((base["X"][control_ix]-mean)/std).astype(np.float32))
    rows=[]
    for name,layer in layers.items():
        patched,_=run_with_patch(model,Xtarget,layer,ref[name]); dist=np.linalg.norm(patched-prototype,axis=1); rows.append({"intervention":name,"reference_user":TARGET,"n_target_windows":len(dist),"baseline_mean_distance":float(baseline_dist.mean()),"patched_mean_distance":float(dist.mean()),"distance_recovery":float(baseline_dist.mean()-dist.mean()),"recovery_fraction":float((baseline_dist.mean()-dist.mean())/baseline_dist.mean()),"patched_frr_relative_to_baseline":float(np.mean(dist>1.1885696649551392))})
        wrong,_=run_with_patch(model,Xtarget,layer,ref[name]); # overwritten below with explicit wrong reference
        wrong_ref=collect_reference(model,Xcontrol,layer); wrong_z,_=run_with_patch(model,Xtarget,layer,wrong_ref); wrong_dist=np.linalg.norm(wrong_z-prototype,axis=1)
        rows[-1]["wrong_user_control_distance"] = float(wrong_dist.mean())
    out=pd.DataFrame(rows); out.to_csv(RESULTS/"causal_trace_user_163748.csv",index=False)
    best=out.sort_values("recovery_fraction",ascending=False).iloc[0]
    summary={"target_user":TARGET,"stage":STAGE,"reference":"target user's first-30-day train activations","control_reference_user":control_user,"baseline_mean_distance":float(baseline_dist.mean()),"baseline_distance_std":float(baseline_dist.std()),"best_intervention":str(best.intervention),"best_recovery_fraction":float(best.recovery_fraction),"best_distance_recovery":float(best.distance_recovery),"interpretation":"Activation patching is an intervention-based localization test; it indicates which representation can restore the target user's early embedding behavior, not that a single layer literally stores the whole user identity."}
    (RESULTS/"causal_trace_user_163748_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plot=out.copy(); x=np.arange(len(plot)); w=.25; fig,ax=plt.subplots(figsize=(11,6)); ax.bar(x-w,plot.baseline_mean_distance,width=w,label="Unpatched Stage 2",color="#999999"); ax.bar(x,plot.patched_mean_distance,width=w,label="Patched with user 163748",color="#386cb0"); ax.bar(x+w,plot.wrong_user_control_distance,width=w,label="Wrong-user control",color="#f0027f"); ax.set_xticks(x,plot.intervention,rotation=25,ha="right"); ax.set_ylabel("Mean embedding distance to user 163748 prototype"); ax.set_title("Causal tracing by activation patching: user 163748"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"causal_trace_user_163748.png",dpi=200); plt.close(fig)
    print(json.dumps(summary,indent=2)); print(out.to_string(index=False))

if __name__=="__main__": main()
