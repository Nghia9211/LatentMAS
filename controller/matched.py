"""Historical-style batch16 data contract and true per-row online stopping.

Official train is the only training source. Historical test is a separately
called evaluation selector. No shrinking active batches: finished rows become
masked padding; physical CUDA work persists until the last active row stops.
"""
import json
from pathlib import Path
import torch
from torch.distributions import Categorical
from controller.profile import load_official_medqa,make_subset,normalized_question,raw_observation,file_hash
from latent_batch import rollout

ROOT = Path(__file__).resolve().parents[1]


def historical_style(item):
    stem=item['question_stem'].strip()
    options=item['question'].rsplit('\n\n',1)[-1]
    return stem+'\n'+options+'\nChoose the correct option.'


def train_items(data_dir,count=32,seed=42):
    if count!=32:raise ValueError('Matched pilot requires32 official train questions')
    pool,meta=load_official_medqa(data_dir,'train')
    selected=[dict(item,question=historical_style(item)) for item in make_subset(pool,count,seed)]
    # Audit held-out stem hashes only; never return held-out prompts/golds/features.
    forbidden=set()
    for split in ('validation','test'):
        items,_=load_official_medqa(data_dir,split)
        forbidden.update(normalized_question(i['question_stem']) for i in items)
    if any(i['source_split']!='train' or ':train:' not in i['source_id'] or normalized_question(i['question_stem']) in forbidden for i in selected):
        raise ValueError('Train split/stem firewall failed')
    return selected,dict(split='train',count=32,dataset_sha256=meta['sha256'],
        source_ids=[i['source_id'] for i in selected],seed=seed,heldout_overlap=False,
        heldout_questions_used_as_training_examples=False)


def require_train(items):
    if len(items)!=32 or any(i.get('source_split')!='train' or ':train:' not in i.get('source_id','') for i in items):
        raise ValueError('Official train32 only; test/validation examples forbidden here')


def select_test16(data_dir):
    historical=json.loads((ROOT/'data/medqa.json').read_text())
    items,_=load_official_medqa(data_dir,'test')
    lookup={normalized_question(item['question_stem']):item for item in items}
    selected=[]
    for index,row in enumerate(historical[:16]):
        stem=row['question'].rstrip();suffix='\n'.join(row['options'])
        while stem.endswith(suffix):stem=stem[:-len(suffix)].rstrip()
        item=dict(lookup[normalized_question(stem)])
        if item['question'].split('\n\n')[-1].splitlines()!=row['options']:
            raise ValueError('Historical/official options differ')
        option=next(o[3:] for o in row['options'] if o[:1].lower()==item['gold'])
        if option!=row['answer']:raise ValueError('Historical/official gold differ')
        item['historical_index']=index;selected.append(item)
    if len(selected)!=16 or len({item['source_id'] for item in selected})!=16:
        raise ValueError('Need16 distinct matched test identities')
    train,_=load_official_medqa(data_dir,'train')
    train_stems={normalized_question(item['question_stem']) for item in train}
    if any(normalized_question(item['question_stem']) in train_stems for item in selected):
        raise ValueError('Train/test stem overlap')
    return selected


def test_items(data_dir,reference):
    items=select_test16(data_dir)
    rows=json.loads(Path(reference).read_text())['predictions'][:16]
    for item,row in zip(items,rows):
        expected=historical_style(item)
        if expected!=row['question'] or item['gold']!=row['gold']:
            raise ValueError('Reference question/gold differs from official test identity')
        item['question']=expected
    return items


def render_batch(wrapper,args,items):
    from prompts import build_agent_message_sequential_latent_mas
    texts={key:[wrapper.render_chat(build_agent_message_sequential_latent_mas(
        role=role,question=i['question'],context='',method='latent_mas',args=args),add_generation_prompt=True)
        for i in items] for key,role in [('A','planner'),('B','judger')]}
    encoded={key:wrapper.tokenizer(value,return_tensors='pt',padding=True,add_special_tokens=False)
        for key,value in texts.items()}
    return texts,encoded


class CohortDecisions:
    def __init__(self,model,batch,device,stochastic):
        self.model=model;self.stochastic=stochastic
        self.histories=[[] for _ in range(batch)]
        dim=model.encoder[0].in_features
        self.x=torch.zeros(batch,7,dim,device=device)
        self.actions=torch.zeros(batch,7,dtype=torch.long,device=device)
        self.visited=torch.zeros(batch,7,dtype=torch.bool,device=device)
        self.logprobs=torch.zeros(batch,7,device=device)
        self.values=torch.zeros(batch,7,device=device)
        self.budgets=torch.full((batch,),80,dtype=torch.long,device=device)

    def begin(self,h0):
        for i,h in enumerate(h0):self.histories[i].append(h.detach().clone())

    @torch.no_grad()
    def step(self,k,hidden,active):
        indices=torch.where(active)[0]
        for i in indices.tolist():self.histories[i].append(hidden[i].detach().clone())
        stopped=torch.zeros_like(active)
        if k%10:return stopped
        if k==80:return active.clone()
        obs=torch.stack([raw_observation(torch.stack(self.histories[i]),k) for i in indices.tolist()])
        logits,values=self.model(obs)
        distribution=Categorical(logits=logits)
        actions=distribution.sample() if self.stochastic else logits.argmax(-1)
        t=k//10-1
        self.x[indices,t]=obs;self.actions[indices,t]=actions
        self.visited[indices,t]=True;self.logprobs[indices,t]=distribution.log_prob(actions)
        self.values[indices,t]=values
        stopping=indices[actions==0];stopped[stopping]=True;self.budgets[stopping]=k
        return stopped


@torch.no_grad()
def online_rollout(wrapper,ids,mask,decisions,max_steps=80,preserve_arithmetic=False):
    """Keep batch shape fixed, but mask/freeze stopped rows and stop whole cohort.

    Prefixes contain only real A prompt plus actual e1..eK for that row. Full
    cache retains masked dummy columns; no active compaction numerical change.
    """
    h0=[]
    def initial_observer(**kw):
        h0.append(kw['hidden_last_position'][:,-1].detach().clone())
    past,embeddings=rollout(wrapper,ids,mask,latent_steps=0,observer=initial_observer)
    hidden=h0[0];decisions.begin(hidden)
    active=torch.ones(len(ids),dtype=torch.bool,device=ids.device)
    full_mask=past._latent_attention_mask
    positions=past._latent_next_position
    model=getattr(wrapper,'HF_model',None)
    if model is None:model=wrapper.model
    for step in range(1,max_steps+1):
        prior=active.clone()
        latent=wrapper._apply_latent_realignment(hidden,model).unsqueeze(1)
        if not preserve_arithmetic:latent=latent.masked_fill(~prior[:,None,None],0)
        physical=torch.ones_like(prior) if preserve_arithmetic else prior
        full_mask=torch.cat([full_mask,physical[:,None].long()],dim=1)
        out=model(inputs_embeds=latent,attention_mask=full_mask,position_ids=positions[:,None],
            past_key_values=past,use_cache=True,output_hidden_states=True,return_dict=True)
        past=out.past_key_values
        hidden=out.hidden_states[-1][:,-1] if preserve_arithmetic else torch.where(prior[:,None],out.hidden_states[-1][:,-1],hidden)
        for i in torch.where(prior)[0].tolist():embeddings[i]=torch.cat([embeddings[i],latent[i]],dim=0)
        positions=positions+physical.long()
        stopping=decisions.step(step,hidden,prior)
        if stopping.dtype!=torch.bool or stopping.shape!=prior.shape or (stopping&~prior).any():
            raise ValueError('Malformed per-row decisions')
        active=prior&~stopping
        if not active.any():break
    if active.any():raise ValueError('Did not terminate at cap')
    past._latent_attention_mask=full_mask;past._latent_next_position=positions
    past._controller_selected_budgets=decisions.budgets.detach().clone()
    past._controller_physical_steps=step
    return past,embeddings,step


def prefixes_for_b(wrapper,texts,encoded,embeddings):
    from latent_batch import insert_prefix
    b=encoded['B'];ids=b['input_ids'].to(wrapper.HF_device)
    with torch.no_grad():embed=wrapper.embedding_layer(ids)
    marker='<|im_start|>user\n';positions=[]
    for text in texts['B']:
        index=text.find(marker)
        if index<0:raise ValueError('Missing Qwen insertion marker')
        positions.append(len(wrapper.tokenizer(text[:index+len(marker)],add_special_tokens=False)['input_ids']))
    return insert_prefix(embed,b['attention_mask'].to(wrapper.HF_device),embeddings,positions)


def reference_prompt_audit(wrapper,args,data_dir,reference):
    # Evaluation-only protocol check; never called by train item selector/optimizer.
    items=test_items(data_dir,reference);texts,encoded=render_batch(wrapper,args,items)
    rows=json.loads(Path(reference).read_text())['predictions'][:16]
    for key,index in [('A',0),('B',1)]:
        for i,row in enumerate(rows):
            if texts[key][i]!=row['agents'][index]['input']:
                raise ValueError(f'Historical {key} rendered prompt mismatch at{i}')
            if 'input_ids' in row['agents'][index]:
                ids=encoded[key]['input_ids'][i,encoded[key]['attention_mask'][i].bool()].tolist()
                if ids!=row['agents'][index]['input_ids']:raise ValueError('Historical token IDs differ')
    return dict(passed=True,cases=16,A_B_rendered_strings_exact=True,
        reference_sha256=file_hash(reference),scope='Protocol auditing only; no test outputs/labels supplied to training')
