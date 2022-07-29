from subprocess import Popen, PIPE, CalledProcessError
import json
from text.torchtext.datasets.generic import Query
import logging
import os
import re
import string
import numpy as np
import collections
from multiprocessing import Pool, cpu_count
from contextlib import closing
from sklearn import metrics
from pyrouge import Rouge155
from sacrebleu import corpus_bleu
import pdb


def to_lf(s, table):
    aggs = [y.lower() for y in Query.agg_ops]
    agg_to_idx = {x: i for i, x in enumerate(aggs)}
    conditionals = [y.lower() for y in Query.cond_ops]
    headers_unsorted = [(y.lower(), i) for i, y in enumerate(table['header'])]
    headers = [(y.lower(), i) for i, y in enumerate(table['header'])]
    headers.sort(reverse=True, key=lambda x: len(x[0]))
    condition_s, conds = None, []
    if 'where' in s:
        s, condition_s = s.split('where', 1)

    s = ' '.join(s.split()[1:-2])
    sel, agg = None, 0
    for col, idx in headers:
        if col == s:
            sel = idx
    if sel is None:
        s = s.split()
        agg = agg_to_idx[s[0]]
        s = ' '.join(s[1:])
        for col, idx in headers:
            if col == s:
                sel = idx

    full_conditions = []
    if not condition_s is None:

        condition_s = ' ' + condition_s + ' '
        for idx, col in enumerate(headers):
            condition_s = condition_s.replace(' ' + col[0] + ' ', ' Col{} '.format(col[1]))
        condition_s = condition_s.strip()

        for idx, col in enumerate(conditionals):
            new_s = []
            for t in condition_s.split():
                if t == col:
                    new_s.append('Cond{}'.format(idx))
                else:
                    new_s.append(t)
            condition_s = ' '.join(new_s)
        s = condition_s
        conds = re.split('(Col\d+ Cond\d+)', s)
        if len(conds) == 0:
            conds = [s]
        conds = [x for x in conds if len(x.strip()) > 0]
        full_conditions = []
        for i, x in enumerate(conds):
            if i % 2 == 0:
                x = x.split()
                col_num = int(x[0].replace('Col', ''))
                opp_num = int(x[1].replace('Cond', ''))
                full_conditions.append([col_num, opp_num])
            else:
                x = x.split()
                if x[-1] == 'and':
                    x = x[:-1]
                x = ' '.join(x)
                if 'Col' in x:
                    new_x = []
                    for t in x.split():
                        if 'Col' in t:
                            idx = int(t.replace('Col', ''))
                            t = headers_unsorted[idx][0]
                        new_x.append(t)
                    x = new_x
                    x = ' '.join(x)
                if 'Cond' in x:
                    new_x = []
                    for t in x.split():
                        if 'Cond' in t:
                            idx = int(t.replace('Cond', ''))
                            t = conditionals[idx]
                        new_x.append(t)
                    x = new_x
                    x = ' '.join(x)
                full_conditions[-1].append(x)
    logical_form = {'sel': sel, 'conds': full_conditions, 'agg': agg}
    return logical_form


def computeLFEM(greedy, answer, args):
    answer = [x[0] for x in answer]
    count = 0
    correct = 0
    text_answers = []
    for idx, (g, ex) in enumerate(zip(greedy, answer)):
        count += 1
        text_answers.append([ex['answer'].lower()])
        try:
            lf = to_lf(g, ex['table'])
            gt = ex['sql']
            conds = gt['conds']
            lower_conds = []
            for c in conds:
                lc = c
                lc[2] = str(lc[2]).lower()
                lower_conds.append(lc)
            gt['conds'] = lower_conds
            correct += lf == gt
        except Exception as e:
            continue
    return correct / count * 100, text_answers


def score(answer, gold):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    tp, tn, sys_pos, real_pos = 0, 0, 0, 0
    if answer == gold:
        if not ('unanswerable' in gold and len(gold) == 1):
            tp += 1
        else:
            tn += 1
    if not ('unanswerable' in answer and len(answer) == 1):
        sys_pos += 1
    if not ('unanswerable' in gold and len(gold) == 1):
        real_pos += 1
    return np.array([tp, tn, sys_pos, real_pos])


def simplify(answer):
    return set(''.join(c for c in t if c not in string.punctuation) for t in answer.strip().lower().split()) - {'the',
                                                                                                                'a',
                                                                                                                'an',
                                                                                                                'and',
                                                                                                                ''}


# http://nlp.cs.washington.edu/zeroshot/evaluate.py
def computeCF1(greedy, answer):
    scores = np.zeros(4)
    for g, a in zip(greedy, answer):
        scores += score(g, a)
    tp, tn, sys_pos, real_pos = scores.tolist()
    total = len(answer)
    if tp == 0:
        p = r = f = 0.0
    else:
        p = tp / float(sys_pos)
        r = tp / float(real_pos)
        f = 2 * p * r / (p + r)

    return f * 100, p * 100, r * 100


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # TODO: Modify the metric: deleting repeated words
    # def remove_repeat(text):
    #     pass
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_multi_ones_hot_vector(y, args):
    val_task2dict = {
        'AAPD': {'csir': 0, 'statme': 1, 'quantph': 2, 'csit': 3, 'mathit': 4, 'statap': 5, 'cscv': 6, 'cscl': 7,
                 'csai': 8,
                 'mathna': 9, 'csms': 10, 'cscr': 11, 'csse': 12, 'cslg': 13, 'csni': 14, 'cssy': 15, 'csds': 16,
                 'cscc': 17,
                 'csfl': 18, 'csro': 19, 'mathoc': 20, 'csma': 21, 'cspf': 22, 'cssi': 23, 'physicssocph': 24,
                 'csdc': 25,
                 'csdb': 26, 'mathco': 27, 'statml': 28, 'mathpr': 29, 'csne': 30, 'csdm': 31, 'condmatstatmech': 32,
                 'cslo': 33,
                 'cscy': 34, 'condmatdisnn': 35, 'csna': 36, 'csce': 37, 'cssc': 38, 'csgt': 39, 'cshc': 40,
                 'qbioqm': 41,
                 'csdl': 42, 'qbionc': 43, 'cscg': 44, 'cmplg': 45, 'cspl': 46, 'mathlo': 47, 'mathnt': 48, 'csmm': 49,
                 'mathst': 50, 'statth': 51, 'nlinao': 52, 'physicsdataan': 53},
        'AAPDEnhance': {'csir': 0, 'statme': 1, 'quantph': 2, 'csit': 3, 'mathit': 4, 'statap': 5, 'cscv': 6, 'cscl': 7,
                        'csai': 8,
                        'mathna': 9, 'csms': 10, 'cscr': 11, 'csse': 12, 'cslg': 13, 'csni': 14, 'cssy': 15, 'csds': 16,
                        'cscc': 17,
                        'csfl': 18, 'csro': 19, 'mathoc': 20, 'csma': 21, 'cspf': 22, 'cssi': 23, 'physicssocph': 24,
                        'csdc': 25,
                        'csdb': 26, 'mathco': 27, 'statml': 28, 'mathpr': 29, 'csne': 30, 'csdm': 31,
                        'condmatstatmech': 32,
                        'cslo': 33,
                        'cscy': 34, 'condmatdisnn': 35, 'csna': 36, 'csce': 37, 'cssc': 38, 'csgt': 39, 'cshc': 40,
                        'qbioqm': 41,
                        'csdl': 42, 'qbionc': 43, 'cscg': 44, 'cmplg': 45, 'cspl': 46, 'mathlo': 47, 'mathnt': 48,
                        'csmm': 49,
                        'mathst': 50, 'statth': 51, 'nlinao': 52, 'physicsdataan': 53},
        'AAPDOri': {'csir': 0, 'statme': 1, 'quantph': 2, 'csit': 3, 'mathit': 4, 'statap': 5, 'cscv': 6, 'cscl': 7,
                    'csai': 8,
                    'mathna': 9, 'csms': 10, 'cscr': 11, 'csse': 12, 'cslg': 13, 'csni': 14, 'cssy': 15, 'csds': 16,
                    'cscc': 17,
                    'csfl': 18, 'csro': 19, 'mathoc': 20, 'csma': 21, 'cspf': 22, 'cssi': 23, 'physicssocph': 24,
                    'csdc': 25,
                    'csdb': 26, 'mathco': 27, 'statml': 28, 'mathpr': 29, 'csne': 30, 'csdm': 31, 'condmatstatmech': 32,
                    'cslo': 33,
                    'cscy': 34, 'condmatdisnn': 35, 'csna': 36, 'csce': 37, 'cssc': 38, 'csgt': 39, 'cshc': 40,
                    'qbioqm': 41,
                    'csdl': 42, 'qbionc': 43, 'cscg': 44, 'cmplg': 45, 'cspl': 46, 'mathlo': 47, 'mathnt': 48,
                    'csmm': 49,
                    'mathst': 50, 'statth': 51, 'nlinao': 52, 'physicsdataan': 53},
        'AAPDDoc': {'er': 0, 'ed': 1, 'ey': 2, 'ew': 3, 'fl': 4, 'ee': 5, 'fj': 6, 'ft': 7, 'ec': 8, 'fr': 9, 'fd': 10,
                    'fg': 11, 'eq': 12, 'fv': 13, 'dy': 14, 'et': 15, 'fx': 16, 'fu': 17, 'ei': 18, 'eh': 19, 'eg': 20,
                    'dx': 21, 'fk': 22, 'eu': 23, 'dz': 24, 'ek': 25, 'fw': 26, 'ej': 27, 'ef': 28, 'fo': 29, 'fa': 30,
                    'fc': 31, 'dw': 32, 'en': 33, 'ea': 34, 'ex': 35, 'fp': 36, 'ff': 37, 'fn': 38, 'ep': 39, 'eb': 40,
                    'fq': 41, 'fh': 42, 'es': 43, 'ev': 44, 'fb': 45, 'el': 46, 'em': 47, 'ez': 48, 'fm': 49, 'fi': 50,
                    'fe': 51, 'eo': 52, 'fs': 53},
        'WOS46985': {"111120": 0, "111104": 1, "11131": 2, "11181": 3, "11153": 4, "11143": 5, "11192": 6, "11142": 7,
                     "11190": 8, "111100": 9, "111126": 10, "1116": 11, "11149": 12, "111107": 13, "11144": 14,
                     "11126": 15,
                     "1115": 16, "111116": 17, "1119": 18, "111121": 19, "11164": 20, "11191": 21, "11151": 22,
                     "11125": 23,
                     "11180": 24, "11113": 25, "11186": 26, "11178": 27, "11199": 28, "111111": 29, "111108": 30,
                     "111119": 31, "11117": 32, "111128": 33, "111113": 34, "11185": 35, "111106": 36, "11189": 37,
                     "11146": 38, "111102": 39, "11130": 40, "11198": 41, "11127": 42, "11116": 43, "11134": 44,
                     "11183": 45,
                     "111130": 46, "11179": 47, "11158": 48, "111105": 49, "11111": 50, "11173": 51, "11177": 52,
                     "111133": 53, "11150": 54, "111103": 55, "11129": 56, "11162": 57, "11161": 58, "111110": 59,
                     "11136": 60, "11135": 61, "11137": 62, "11119": 63, "1117": 64, "11120": 65, "1113": 66,
                     "11115": 67,
                     "111112": 68, "11147": 69, "11154": 70, "11123": 71, "11169": 72, "11196": 73, "11160": 74,
                     "11159": 75,
                     "111122": 76, "11114": 77, "11157": 78, "11165": 79, "111101": 80, "11110": 81, "11118": 82,
                     "111131": 83, "11195": 84, "11139": 85, "11155": 86, "11168": 87, "1111": 88, "111129": 89,
                     "11148": 90,
                     "11182": 91, "11112": 92, "11188": 93, "11170": 94, "111123": 95, "11166": 96, "1114": 97,
                     "11175": 98,
                     "111117": 99, "11156": 100, "11138": 101, "111124": 102, "111118": 103, "1110": 104, "11128": 105,
                     "11197": 106, "11124": 107, "11121": 108, "111115": 109, "11171": 110, "11133": 111, "1112": 112,
                     "111127": 113, "11141": 114, "111109": 115, "11187": 116, "111114": 117, "11176": 118,
                     "11122": 119,
                     "11167": 120, "11174": 121, "11184": 122, "11145": 123, "11172": 124, "111132": 125, "11193": 126,
                     "111125": 127, "11152": 128, "11163": 129, "11194": 130, "1118": 131, "11132": 132, "11140": 133},
        'WOS11967': {"11118": 0, "1113": 1, "11122": 2, "11121": 3, "11123": 4, "11113": 5, "1111": 6, "11111": 7,
                     "1114": 8, "11128": 9, "1115": 10, "1116": 11, "11131": 12, "11125": 13, "11110": 14, "1118": 15,
                     "11117": 16, "1110": 17, "11116": 18, "11127": 19, "11119": 20, "11114": 21, "11132": 22,
                     "11115": 23, "11126": 24, "1112": 25, "11130": 26, "11129": 27, "11124": 28, "1119": 29,
                     "11112": 30, "11120": 31, "1117": 32},
        'WOS5736': {"1119": 0, "1117": 1, "1110": 2, "1115": 3, "1116": 4, "1118": 5, "11110": 6, "1114": 7, "1112": 8,
                    "1111": 9, "1113": 10},
        '20news': {'recsporthockey': 0, 'recmotorcycles': 1, 'talkreligionmisc': 2, 'socreligionchristian': 3,
                   'talkpoliticsmideast': 4, 'compsysmachardware': 5, 'scimed': 6, 'miscforsale': 7, 'altatheism': 8,
                   'compsysibmpchardware': 9, 'compgraphics': 10, 'composmswindowsmisc': 11, 'recautos': 12,
                   'recsportbaseball': 13, 'scicrypt': 14, 'scielectronics': 15, 'scispace': 16, 'compwindowsx': 17,
                   'talkpoliticsguns': 18, 'talkpoliticsmisc': 19},
        'TREC6': {'abbr': 0, 'loc': 1, 'num': 2, 'hum': 3, 'enty': 4, 'desc': 5},
        'TREC50': {'country': 0, 'animal': 1, 'ord': 2, 'abb': 3, 'state': 4, 'word': 5, 'volsize': 6, 'temp': 7,
                   'religion': 8, 'symbol': 9, 'substance': 10, 'manner': 11, 'city': 12, 'exp': 13, 'termeq': 14,
                   'letter': 15, 'weight': 16, 'code': 17, 'sport': 18, 'other': 19, 'money': 20, 'count': 21,
                   'food': 22, 'plant': 23, 'reason': 24, 'color': 25, 'product': 26, 'period': 27, 'gr': 28,
                   'currency': 29, 'title': 30, 'dismed': 31, 'speed': 32, 'cremat': 33, 'desc': 34, 'perc': 35,
                   'date': 36, 'body': 37, 'instru': 38, 'mount': 39, 'veh': 40, 'techmeth': 41, 'def': 42, 'event': 43,
                   'ind': 44, 'lang': 45, 'dist': 46},
        'OhsumedSingle': {'11113': 0, '11112': 1, '11115': 2, '11117': 3, '11103': 4, '11118': 5, '11119': 6,
                          '11123': 7, '11120': 8, '11105': 9, '11109': 10, '11104': 11, '11121': 12, '11116': 13,
                          '11101': 14, '11107': 15, '11102': 16, '11108': 17, '11106': 18, '11111': 19, '11114': 20,
                          '11122': 21, '11110': 22},
        'OhsumedMulti': {'11113': 0, '11112': 1, '11115': 2, '11117': 3, '11103': 4, '11118': 5, '11119': 6, '11123': 7,
                         '11120': 8, '11105': 9, '11109': 10, '11104': 11, '11121': 12, '11116': 13, '11101': 14,
                         '11107': 15, '11102': 16, '11108': 17, '11106': 18, '11111': 19, '11114': 20, '11122': 21,
                         '11110': 22},
        'YahooAnswers': {'societyculture': 0, 'sciencemathematics': 1, 'health': 2, 'educationreference': 3,
                         'computersinternet': 4, 'sports': 5, 'businessfinance': 6, 'entertainmentmusic': 7,
                         'familyrelationships': 8, 'politicsgovernment': 9},
        'Reuters90': {'sugar': 0, 'bop': 1, 'moneyfx': 2, 'dlr': 3, 'carcass': 4, 'reserves': 5, 'wheat': 6,
                      'interest': 7, 'income': 8, 'mealfeed': 9, 'nkr': 10, 'sunoil': 11, 'natgas': 12, 'petchem': 13,
                      'cottonoil': 14, 'fuel': 15, 'retail': 16, 'tea': 17, 'rye': 18, 'jet': 19, 'sunseed': 20,
                      'cocoa': 21, 'dmk': 22, 'jobs': 23, 'grain': 24, 'corn': 25, 'cpu': 26, 'hog': 27, 'soyoil': 28,
                      'coconut': 29, 'palmoil': 30, 'rand': 31, 'heat': 32, 'lei': 33, 'oilseed': 34, 'rapeseed': 35,
                      'lead': 36, 'orange': 37, 'trade': 38, 'ironsteel': 39, 'silver': 40, 'oat': 41, 'castoroil': 42,
                      'crude': 43, 'strategicmetal': 44, 'coconutoil': 45, 'instaldebt': 46, 'propane': 47, 'nzdlr': 48,
                      'coffee': 49, 'naphtha': 50, 'cpi': 51, 'gold': 52, 'palladium': 53, 'nickel': 54, 'dfl': 55,
                      'copper': 56, 'rice': 57, 'groundnutoil': 58, 'rubber': 59, 'platinum': 60, 'ipi': 61, 'zinc': 62,
                      'palmkernel': 63, 'linoil': 64, 'acq': 65, 'gas': 66, 'soymeal': 67, 'tin': 68, 'soybean': 69,
                      'ship': 70, 'potato': 71, 'unknown': 72, 'groundnut': 73, 'vegoil': 74, 'moneysupply': 75,
                      'cotton': 76, 'livestock': 77, 'yen': 78, 'alum': 79, 'sunmeal': 80, 'earn': 81, 'sorghum': 82,
                      'lumber': 83, 'barley': 84, 'gnp': 85, 'lcattle': 86, 'wpi': 87, 'housing': 88, 'copracake': 89,
                      'rapeoil': 90},
        'Reuters115': {'dfl': 0, 'rice': 1, 'silver': 2, 'nkr': 3, 'sunseed': 4, 'nickel': 5, 'dkr': 6, 'oat': 7,
                       'plywood': 8, 'lead': 9, 'rubber': 10, 'fuel': 11, 'naphtha': 12, 'rye': 13, 'linmeal': 14,
                       'housing': 15, 'wpi': 16, 'instaldebt': 17, 'palmkernel': 18, 'ringgit': 19, 'tea': 20,
                       'heat': 21, 'inventories': 22, 'unknown': 23, 'alum': 24, 'sunmeal': 25, 'orange': 26,
                       'potato': 27, 'soyoil': 28, 'ship': 29, 'coconutoil': 30, 'jet': 31, 'copper': 32, 'grain': 33,
                       'crude': 34, 'palmoil': 35, 'cocoa': 36, 'copracake': 37, 'castoroil': 38, 'propane': 39,
                       'saudriyal': 40, 'gas': 41, 'moneysupply': 42, 'cornglutenfeed': 43, 'cpi': 44, 'sorghum': 45,
                       'earn': 46, 'platinum': 47, 'soybean': 48, 'carcass': 49, 'reserves': 50, 'livestock': 51,
                       'natgas': 52, 'soymeal': 53, 'zinc': 54, 'corn': 55, 'palladium': 56, 'cruzado': 57, 'cpu': 58,
                       'bop': 59, 'gold': 60, 'ironsteel': 61, 'income': 62, 'acq': 63, 'ipi': 64, 'gnp': 65,
                       'cottonoil': 66, 'yen': 67, 'linoil': 68, 'petchem': 69, 'lcattle': 70, 'sunoil': 71, 'stg': 72,
                       'tapioca': 73, 'rapeoil': 74, 'fishmeal': 75, 'groundnutoil': 76, 'oilseed': 77, 'skr': 78,
                       'dlr': 79, 'hog': 80, 'linseed': 81, 'dmk': 82, 'can': 83, 'castorseed': 84, 'sugar': 85,
                       'rupiah': 86, 'rapemeal': 87, 'retail': 88, 'lit': 89, 'peseta': 90, 'wool': 91, 'groundnut': 92,
                       'lei': 93, 'rapeseed': 94, 'vegoil': 95, 'mealfeed': 96, 'citruspulp': 97, 'interest': 98,
                       'porkbelly': 99, 'coconut': 100, 'tin': 101, 'lumber': 102, 'trade': 103, 'austdlr': 104,
                       'cornoil': 105, 'rand': 106, 'coffee': 107, 'cotton': 108, 'jobs': 109, 'nzdlr': 110,
                       'wheat': 111, 'moneyfx': 112, 'strategicmetal': 113, 'barley': 114, 'redbean': 115},
        '20newsSingle': {'scimed': 0, 'socreligionchristian': 1, 'scielectronics': 2, 'recautos': 3,
                         'talkreligionmisc': 4, 'miscforsale': 5, 'compgraphics': 6, 'composmswindowsmisc': 7,
                         'recmotorcycles': 8, 'compsysibmpchardware': 9, 'altatheism': 10, 'recsporthockey': 11,
                         'talkpoliticsguns': 12, 'recsportbaseball': 13, 'talkpoliticsmideast': 14,
                         'talkpoliticsmisc': 15, 'compwindowsx': 16, 'scispace': 17, 'compsysmachardware': 18,
                         'scicrypt': 19},
        '20newsMulti': {'scimed': 0, 'socreligionchristian': 1, 'scielectronics': 2, 'recautos': 3,
                        'talkreligionmisc': 4, 'miscforsale': 5, 'compgraphics': 6, 'composmswindowsmisc': 7,
                        'recmotorcycles': 8, 'compsysibmpchardware': 9, 'altatheism': 10, 'recsporthockey': 11,
                        'talkpoliticsguns': 12, 'recsportbaseball': 13, 'talkpoliticsmideast': 14,
                        'talkpoliticsmisc': 15, 'compwindowsx': 16, 'scispace': 17, 'compsysmachardware': 18,
                        'scicrypt': 19},
        'AmazonFullReview': {'clsone': 0, 'clstwo': 1, 'clsthree': 2, 'clsfour': 3, 'clsfive': 4},
        'R8': {'acq': 0, 'crude': 1, 'earn': 2, 'grain': 3, 'interest': 4, 'moneyfx': 5, 'ship': 6, 'trade': 7},
        'R52': {'acq': 0, 'alum': 1, 'bop': 2, 'carcass': 3, 'cocoa': 4, 'coffee': 5, 'copper': 6, 'cotton': 7,
                'cpi': 8, 'cpu': 9, 'crude': 10, 'dlr': 11, 'earn': 12, 'fuel': 13, 'gas': 14, 'gnp': 15, 'gold': 16,
                'grain': 17, 'heat': 18, 'housing': 19, 'income': 20, 'instaldebt': 21, 'interest': 22, 'ipi': 23,
                'ironsteel': 24, 'jet': 25, 'jobs': 26, 'lead': 27, 'lei': 28, 'livestock': 29, 'lumber': 30,
                'mealfeed': 31, 'moneyfx': 32, 'moneysupply': 33, 'natgas': 34, 'nickel': 35, 'orange': 36,
                'petchem': 37, 'platinum': 38, 'potato': 39, 'reserves': 40, 'retail': 41, 'rubber': 42, 'ship': 43,
                'strategicmetal': 44, 'sugar': 45, 'tea': 46, 'tin': 47, 'trade': 48, 'vegoil': 49, 'wpi': 50,
                'zinc': 51},
        'YelpFullReview': {'clsone': 0, 'clstwo': 1, 'clsthree': 2, 'clsfour': 3, 'clsfive': 4},
    }

    assert len(args.val_tasks) == 1, 'for now only support single task micro f1 metric'
    l2i_dict = val_task2dict[args.val_tasks[0]]

    y = y.split()
    # ground_truth_tokens = ground_truth.split()

    total_length = len(l2i_dict)
    ones_hot = np.zeros(total_length, dtype=np.int)
    # true_ones_hot = np.zeros(total_length, dtype=np.int)
    hot_indices = [l2i_dict[token] for token in y if token in l2i_dict.keys()]
    # true_hot_indices = [l2i_dict[token] for token in ground_truth_tokens]
    ones_hot[hot_indices] = 1
    # true_ones_hot[true_hot_indices] = 1
    return ones_hot


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(prediction, ground_truth):
    return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def computeMicroF1(outputs, targets, args):
    preds = np.array([compute_multi_ones_hot_vector(output, args) for output in outputs], dtype=np.int)
    tgts = np.array([compute_multi_ones_hot_vector(tgt[0], args) for tgt in targets], dtype=np.int)
    micro_f1 = metrics.f1_score(tgts, preds, average='micro') * 100.0
    precision = metrics.precision_score(tgts, preds, average='micro') * 100.0
    recall = metrics.recall_score(tgts, preds, average='micro') * 100.0
    return micro_f1, precision, recall


def computeF1(outputs, targets):
    # Origin
    return sum([metric_max_over_ground_truths(f1_score, o, t) for o, t in zip(outputs, targets)]) / len(outputs) * 100
    # preds = np.array([compute_multi_ones_hot_vector(output) for output in outputs], dtype=np.int)
    # tgts = np.array([compute_multi_ones_hot_vector(tgt[0]) for tgt in targets], dtype=np.int)
    # micro_f1 = metrics.f1_score(tgts, preds, average='micro') * 100.0
    # return micro_f1


def computeEM(outputs, targets):
    outs = [metric_max_over_ground_truths(exact_match, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score


class Rouge(Rouge155):
    """Rouge calculator class with custom command-line options."""

    # See full list of options here:
    # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/README.txt#L82
    DEFAULT_OPTIONS = [
        '-a',  # evaluate all systems
        '-n', 4,  # max-ngram
        '-x',  # do not calculate ROUGE-L
        '-2', 4,  # max-gap-length
        '-u',  # include unigram in skip-bigram
        '-c', 95,  # confidence interval
        '-r', 1000,  # number-of-samples (for resampling)
        '-f', 'A',  # scoring formula
        '-p', 0.5,  # 0 <= alpha <=1
        '-t', 0,  # count by token instead of sentence
        '-d',  # print per evaluation scores
    ]

    def __init__(self, n_words=None,
                 keep_files=False, options=None):

        if options is None:
            self.options = self.DEFAULT_OPTIONS.copy()
        else:
            self.options = options

        if n_words:
            options.extend(["-l", n_words])

        stem = "-m" in self.options

        super(Rouge, self).__init__(
            n_words=n_words, stem=stem,
            keep_files=keep_files)

    def _run_rouge(self):
        # Get full options
        options = (
                ['-e', self._rouge_data] +
                list(map(str, self.options)) +
                [os.path.join(self._config_dir, "settings.xml")])

        logging.info("Running ROUGE with options {}".format(" ".join(options)))
        # print([self._rouge_bin] + list(options))
        pipes = Popen([self._rouge_bin] + options, stdout=PIPE, stderr=PIPE)
        std_out, std_err = pipes.communicate()

        div_by_zero_error = std_err.decode("utf-8"). \
            startswith("Illegal division by zero")
        if pipes.returncode == 0 or div_by_zero_error:
            # Still returns the correct output even with div by zero
            return std_out
        else:
            raise ValueError(
                std_out.decode("utf-8") + "\n" + std_err.decode("utf-8"))


def computeROUGE(greedy, answer):
    rouges = compute_rouge_scores(greedy, answer)
    if len(rouges) > 0:
        avg_rouges = {}
        for key in rouges[0].keys():
            avg_rouges[key] = sum(
                [r.get(key, 0.0) for r in rouges]) / len(rouges) * 100
    else:
        avg_rouges = None
    return avg_rouges


def split_sentences(txt, splitchar=".", include_splitchar=False):
    """Split sentences of a text based on a given EOS char."""
    out = [s.split() for s in txt.strip().split(splitchar) if len(s) > 0]
    return out


def compute_rouge_scores(summs, refs, splitchar='.', options=None, parallel=True):
    assert len(summs) == len(refs)
    options = [
        '-a',  # evaluate all systems
        '-c', 95,  # confidence interval
        '-m',  # use Porter stemmer
        '-n', 2,  # max-ngram
        '-w', 1.3,  # weight (weighting factor for WLCS)
    ]
    rr = Rouge(options=options)
    rouge_args = []
    for summ, ref in zip(summs, refs):
        letter = "A"
        ref_dict = {}
        for r in ref:
            ref_dict[letter] = [x for x in split_sentences(r, splitchar) if len(x) > 0]
            letter = chr(ord(letter) + 1)
        s = [x for x in split_sentences(summ, splitchar) if len(x) > 0]
        rouge_args.append((s, ref_dict))
    if parallel:
        with closing(Pool(cpu_count() // 2)) as pool:
            rouge_scores = pool.starmap(rr.score_summary, rouge_args)
    else:
        rouge_scores = []
        for s, a in rouge_args:
            rouge_scores.append(rr.score_summary(s, ref_dict))
    return rouge_scores


def to_delta_state(line):
    delta_state = {'inform': {}, 'request': {}}
    try:
        if line == 'None' or line.strip() == '' or line.strip() == ';':
            return delta_state
        inform, request = [[y.strip() for y in x.strip().split(',')] for x in line.split(';')]
        inform_pairs = {}
        for i in inform:
            try:
                k, v = i.split(':')
                inform_pairs[k.strip()] = v.strip()
            except:
                pass
        delta_state = {'inform': inform_pairs, 'request': request}
    except:
        pass
    finally:
        return delta_state


def update_state(state, delta):
    for act, slot in delta.items():
        state[act] = slot
    return state


def dict_cmp(d1, d2):
    def cmp(a, b):
        for k1, v1 in a.items():
            if k1 not in b:
                return False
            else:
                if v1 != b[k1]:
                    return False
        return True

    return cmp(d1, d2) and cmp(d2, d1)


def computeDialogue(greedy, answer):
    examples = []
    for idx, (g, a) in enumerate(zip(greedy, answer)):
        examples.append((a[0][0], g, a[0][1], idx))
    examples.sort()
    turn_request_positives = 0
    turn_goal_positives = 0
    joint_goal_positives = 0
    ldt = None
    for ex in examples:
        if ldt is None or ldt.split('_')[:-1] != ex[0].split('_')[:-1]:
            state, answer_state = {}, {}
            ldt = ex[0]
        delta_state = to_delta_state(ex[1])
        answer_delta_state = to_delta_state(ex[2])
        state = update_state(state, delta_state['inform'])
        answer_state = update_state(answer_state, answer_delta_state['inform'])
        if dict_cmp(state, answer_state):
            joint_goal_positives += 1
        if delta_state['request'] == answer_delta_state['request']:
            turn_request_positives += 1
        if dict_cmp(delta_state['inform'], answer_delta_state['inform']):
            turn_goal_positives += 1

    joint_goal_em = joint_goal_positives / len(examples) * 100
    turn_request_em = turn_request_positives / len(examples) * 100
    turn_goal_em = turn_goal_positives / len(examples) * 100
    answer = [(x[-1], x[-2]) for x in examples]
    answer.sort()
    answer = [[x[1]] for x in answer]
    return joint_goal_em, turn_request_em, turn_goal_em, answer


def compute_metrics(greedy, answer, rouge=False, bleu=False, corpus_f1=False, logical_form=False, args=None,
                    dialogue=False):
    metric_keys = []
    metric_values = []
    if not isinstance(answer[0], list):
        answer = [[a] for a in answer]
    # if logical_form:
    #     lfem, answer = computeLFEM(greedy, answer, args)
    #     metric_keys += ['lfem']
    #     metric_values += [lfem]
    # if dialogue:
    #     joint_goal_em, request_em, turn_goal_em, answer = computeDialogue(greedy, answer)
    #     avg_dialogue = (joint_goal_em + request_em) / 2
    #     metric_keys += ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue']
    #     metric_values += [joint_goal_em, request_em, turn_goal_em, avg_dialogue]
    # em = computeEM(greedy, answer)
    # metric_keys += ['em']
    # metric_values += [em]
    # if bleu:
    #     bleu = computeBLEU(greedy, answer)
    #     metric_keys.append('bleu')
    #     metric_values.append(bleu)
    # if rouge:
    #     rouge = computeROUGE(greedy, answer)
    #     metric_keys += ['rouge1', 'rouge2', 'rougeL', 'avg_rouge']
    #     avg_rouge = (rouge['rouge_1_f_score'] + rouge['rouge_2_f_score'] + rouge['rouge_l_f_score']) / 3
    #     metric_values += [rouge['rouge_1_f_score'], rouge['rouge_2_f_score'], rouge['rouge_l_f_score'], avg_rouge]
    norm_greedy = [normalize_text(g) for g in greedy]
    norm_answer = [[normalize_text(a) for a in al] for al in answer]
    mf1, precision, recall = computeMicroF1(norm_greedy, norm_answer, args)
    # nem = computeEM(norm_greedy, norm_answer)
    metric_keys.extend(['micro-f1', 'precision', 'recall'])
    metric_values.extend([mf1, precision, recall])
    # if corpus_f1:
    #     corpus_f1, precision, recall = computeCF1(norm_greedy, norm_answer)
    #     metric_keys += ['corpus_f1', 'precision', 'recall']
    #     metric_values += [corpus_f1, precision, recall]
    metric_dict = collections.OrderedDict(list(zip(metric_keys, metric_values)))
    return metric_dict, answer
