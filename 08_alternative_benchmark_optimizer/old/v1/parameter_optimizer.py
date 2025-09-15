from datasets import load_dataset
from glob import glob
from tqdm import tqdm
import numpy as np
import math
from fractions import Fraction
from scipy.optimize import minimize
import json

def get_vectors(angles):
    return np.column_stack((np.sin(angles*(np.pi/2)),np.cos(angles*(np.pi/2))))

def get_angles(vectors):
    return np.array([math.atan2(*x) / (np.pi / 2)  for x in vectors])

def convertMedienlandschaft(x):
    return float(-Fraction(4/3)+Fraction(1/3)*x)

v_linke = [-1.0, 0.0]
v_gruene = [-0.9077674463309972, 0.4194737934385177]
v_spd = [-0.8074405688999996, 0.5899489195637577]
v_fdp = [0.0, 1.0]
v_cdu = [0.614758038308033, 0.788715762702673]
v_afd = [1.0, 0.0]

exp = {}
exp['NLP-UniBW/deutschlandfunk_de_classified'] = convertMedienlandschaft(3.8)
exp['NLP-UniBW/focus_de_classified'] = convertMedienlandschaft(4.9)
exp['NLP-UniBW/linksunten_classified'] = convertMedienlandschaft(2.0)
exp['NLP-UniBW/taz_de_classified'] = convertMedienlandschaft(2.8)
exp['NLP-UniBW/zeit_de_classified'] = convertMedienlandschaft(3.6)
exp['NLP-UniBW/stern_de_classified'] = convertMedienlandschaft(3.8)
exp['NLP-UniBW/tichyseinblick_de_classified'] = convertMedienlandschaft(5.5)
exp['NLP-UniBW/cicero_de_classified'] = convertMedienlandschaft(4.9)
exp['NLP-UniBW/spiegel_de_classified'] = convertMedienlandschaft(3.5)
exp['NLP-UniBW/vice_de_classified'] = convertMedienlandschaft(2.8)
exp['NLP-UniBW/tagesschau_de_classified'] = convertMedienlandschaft(3.7)
exp['NLP-UniBW/sueddeutsche_de_classified'] = convertMedienlandschaft(3.5)
exp['NLP-UniBW/welt_de_classified'] = convertMedienlandschaft(4.8)
exp['NLP-UniBW/mdr_de_classified'] = convertMedienlandschaft(4.1)
exp['NLP-UniBW/der_freitag_de_classified'] = convertMedienlandschaft(2.7)
exp['NLP-UniBW/frankfurter_rundschau_de_classified'] = convertMedienlandschaft(3.4)
exp['NLP-UniBW/bild_de_classified'] = convertMedienlandschaft(5.2)
exp['NLP-UniBW/russia_today_de_classified'] = convertMedienlandschaft(5.1)
exp['NLP-UniBW/tagesspiegel_de_classified'] = convertMedienlandschaft(3.6)
exp['NLP-UniBW/br_de_classified'] = convertMedienlandschaft(4.4)
exp['NLP-UniBW/achgut_de_classified'] = convertMedienlandschaft(5.2)
exp['NLP-UniBW/wdr_de_classified'] = convertMedienlandschaft(3.5)
exp['NLP-UniBW/neues_deutschland_de_classified'] = convertMedienlandschaft(2.6)
exp['NLP-UniBW/compact_de_classified'] = convertMedienlandschaft(6.0)
exp['NLP-UniBW/ndr_de_classified'] = convertMedienlandschaft(3.7)
exp['NLP-UniBW/nachdenkseiten_de_classified'] = convertMedienlandschaft(3.1)
exp['NLP-UniBW/junge_freiheit_de_classified'] = convertMedienlandschaft(5.8)
exp['NLP-UniBW/rtl_de_classified'] = convertMedienlandschaft(4.5)
exp['NLP-UniBW/junge_welt_classified'] = convertMedienlandschaft(2.4)
exp['NLP-UniBW/ntv_de_classified'] = convertMedienlandschaft(4.3)
exp['NLP-UniBW/jungle_world_classified'] = convertMedienlandschaft(2.3)
exp['NLP-UniBW/frankfurter_allgemeine_de_classified'] = convertMedienlandschaft(4.5)
exp['NLP-UniBW/mm_news_de_classified'] = convertMedienlandschaft(5.1)

a_linke = math.atan2(*v_linke) / (np.pi / 2)
a_gruene = math.atan2(*v_gruene) / (np.pi / 2)
a_spd = math.atan2(*v_spd) / (np.pi / 2)
a_fdp = math.atan2(*v_fdp) / (np.pi / 2)
a_cdu = math.atan2(*v_cdu) / (np.pi / 2)
a_afd = math.atan2(*v_afd) / (np.pi / 2)

initial_angles = np.array([a_linke, a_gruene, a_spd, a_fdp, a_cdu, a_afd])

df = load_dataset("NLP-UniBW/deutschlandfunk_de_classified", split="train").to_pandas()
l_ref = []
l_classifier = []
for x in list(df.keys()):
    ref = x.replace("_", "-").split("-")[0]
    if ref in l_ref:
        l_classifier.append("_".join(x.split("_")[:-1]))
    l_ref.append(ref)
l_classifier = sorted(list(set(l_classifier)))
l_parties = ['DIE LINKE', 'BÜNDNIS 90/DIE GRÜNEN', 'SPD', 'FDP', 'CDU/CSU', 'AfD']

l_newspapers = [
    'NLP-UniBW/deutschlandfunk_de_classified',
    'NLP-UniBW/focus_de_classified',
    'NLP-UniBW/linksunten_classified',
    'NLP-UniBW/taz_de_classified',
    'NLP-UniBW/zeit_de_classified',
    'NLP-UniBW/stern_de_classified',
    'NLP-UniBW/tichyseinblick_de_classified',
    'NLP-UniBW/cicero_de_classified',
    'NLP-UniBW/spiegel_de_classified',
    'NLP-UniBW/vice_de_classified',
    'NLP-UniBW/tagesschau_de_classified',
    'NLP-UniBW/sueddeutsche_de_classified',
    'NLP-UniBW/welt_de_classified',
    'NLP-UniBW/mdr_de_classified',
    'NLP-UniBW/der_freitag_de_classified',
    'NLP-UniBW/frankfurter_rundschau_de_classified',
    'NLP-UniBW/bild_de_classified',
    'NLP-UniBW/russia_today_de_classified',
    'NLP-UniBW/tagesspiegel_de_classified',
    'NLP-UniBW/br_de_classified',
    'NLP-UniBW/achgut_de_classified',
    'NLP-UniBW/wdr_de_classified',
    'NLP-UniBW/neues_deutschland_de_classified',
    'NLP-UniBW/compact_de_classified',
    'NLP-UniBW/ndr_de_classified',
    'NLP-UniBW/nachdenkseiten_de_classified',
    'NLP-UniBW/junge_freiheit_de_classified',
    'NLP-UniBW/rtl_de_classified',
    'NLP-UniBW/junge_welt_classified',
    'NLP-UniBW/ntv_de_classified',
    'NLP-UniBW/jungle_world_classified',
    'NLP-UniBW/frankfurter_allgemeine_de_classified',
    'NLP-UniBW/mm_news_de_classified',
]

d_newspapers_df = {}
for newspapers in tqdm(l_newspapers):
    df = load_dataset(newspapers, split="train").to_pandas()
    df = df[df['politic']>=0.8].reset_index(drop=True)
    #df = df.head(1000) #reduce size
    d_newspapers_df[newspapers] = df

d_np = {}
for classifier in l_classifier:
    d_np[classifier] = {}
    for newspaper in l_newspapers:
        df = d_newspapers_df[newspaper][[classifier+"_"+x for x in l_parties]]
        d_np[classifier][newspaper] = np.array(df)

def get_mse(angles, classifier):
    vectors = get_vectors(angles)
    mse = 0
    for newspaper in d_np[classifier]:
        classifications = d_np[classifier][newspaper]
        tmp_angle = np.mean([math.atan2(*x)/(np.pi/2) for x in classifications @ vectors])
        mse = mse + (tmp_angle-exp[newspaper])**2
    mse = mse/len(l_newspapers)
    return mse

Nfeval = 1

def callbackF(x):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}'.format(Nfeval, x[0], x[1], x[2], x[3], x[4], x[5]))
    Nfeval += 1

bounds = [(-1.0, 1.0)] * 6

for classifier in tqdm(l_classifier):
    Nfeval = 1
    result = minimize(
        get_mse,
        initial_angles,
        callback=callbackF,
        args=(classifier),
        method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-9, 'disp': True}
    )
    optimized_params = result.x
    final_result = {
                'classifier':classifier,
                'mse':float(result.fun),
                'optimized_vectors':
                  {
                    'linke': float(optimized_params[0]),
                    'gruene': float(optimized_params[1]),
                    'spd': float(optimized_params[2]),
                    'fdp': float(optimized_params[3]),
                    'cdu': float(optimized_params[4]),
                    'afd': float(optimized_params[5]),
                  }
                 }
    
    json_result = json.dumps(final_result)
    with open("results/"+classifier+".json", mode="w", encoding="utf-8") as f:
        f.write(json_result)