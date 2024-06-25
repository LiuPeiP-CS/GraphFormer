
"""
Define constants for Nary task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PRP': 2, 'NN': 3, 'VBZ': 4, '``': 5, '-LRB-': 6, 'JJR': 7, 'LS': 8, 'MD': 9, 'JJ': 10, 'PRP$': 11, 'RB': 12, "''": 13, 'RP': 14, 'VBP': 15, 'IN': 16, 'CD': 17, 'EX': 18, 'WP$': 19, 'NNS': 20, 'WP': 21, 'VBN': 22, 'PDT': 23, ',': 24, '.': 25, 'NNPS': 26, 'JJS': 27, 'NNP': 28, 'TO': 29, 'WDT': 30, 'SYM': 31, 'POS': 32, 'VBG': 33, 'RBS': 34, 'FW': 35, 'VB': 36, 'RBR': 37, 'CC': 38, 'WRB': 39, 'DT': 40, 'VBD': 41, '-RRB-': 42, ':': 43}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'mwe': 2, 'xcomp': 3, 'next': 4, 'prepc': 5, 'amod': 6, 'appos': 7, 'iobj': 8, 'cop': 9, 'hyphen': 10, 'pcomp': 11, 'preconj': 12, 'partmod': 13, 'prep': 14, 'dep': 15, 'csubjpass': 16, 'neg': 17, 'purpcl': 18, 'tmod': 19, 'poss': 20, 'cc': 21, 'complm': 22, 'infmod': 23, 'agent': 24, 'number': 25, 'acomp': 26, 'abbrev': 27, 'punct': 28, 'rel': 29, 'npadvmod': 30, 'possessive': 31, 'det': 32, 'nsubjpass': 33, 'prt': 34, 'nn': 35, 'quantmod': 36, 'advmod': 37, 'aux': 38, 'ccomp': 39, 'pobj': 40, 'parataxis': 41, 'nsubj': 42, 'expl': 43, 'dobj': 44, 'advcl': 45, 'auxpass': 46, 'predet': 47, 'num': 48, 'conj': 49, 'rcmod': 50, 'csubj': 51, 'self': 52}

DIRECTIONS = 1

#表示无关系的标签
NEGATIVE_LABEL = 'None'

#关系标签
LABEL_TO_ID = {'None': 0, 'resistance or non-response': 1, 'sensitivity': 2, 'response': 3, 'resistance': 4}
# LABEL_TO_ID = {'aka': 0,'attack': 1,'belong_to': 2,'end_time': 3,'find': 4,'goal': 5,
#                 'launch': 6,'located': 7,'occur_time': 8,'Release_time': 9,'use': 10}
# LABEL_TO_ID = {'No': 0, 'Yes': 1}
# LABEL_TO_ID = {
#     'None': 0,
#     'head of government': 1,
#     'country': 2,
#     'place of birth': 3,
#     'place of death': 4,
#     'father': 5,
#     'mother': 6,
#     'spouse': 7,
#     'country of citizenship': 8,
#     'continent': 9,
#     'instance of': 10,
#     'head of state': 11,
#     'capital': 12,
#     'official language': 13,
#     'position held': 14,
#     'child': 15,
#     'author': 16,
#     'member of sports team': 17,
#     'director': 18,
#     'screenwriter': 19,
#     'educated at': 20,
#     'composer': 21,
#     'member of political party': 22,
#     'employer': 23,
#     'founded by': 24,
#     'league': 25,
#     'publisher': 26,
#     'owned by': 27,
#     'located in the administrative territorial entity': 28,
#     'genre': 29,
#     'operator': 30,
#     'religion': 31,
#     'contains administrative territorial entity': 32,
#     'follows': 33,
#     'followed by': 34,
#     'headquarters location': 35,
#     'cast member': 36,
#     'producer': 37,
#     'award received': 38,
#     'creator': 39,
#     'parent taxon': 40,
#     'ethnic group': 41,
#     'performer': 42,
#     'manufacturer': 43,
#     'developer': 44,
#     'series': 45,
#     'sister city': 46,
#     'legislative body': 47,
#     'basin country': 48,
#     'located in or next to body of water': 49,
#     'military branch': 50,
#     'record label': 51,
#     'production company': 52,
#     'location': 53,
#     'subclass of': 54,
#     'subsidiary': 55,
#     'part of': 56,
#     'original language of work': 57,
#     'platform': 58,
#     'mouth of the watercourse': 59,
#     'original network': 60,
#     'member of': 61,
#     'chairperson': 62,
#     'country of origin': 63,
#     'has part': 64,
#     'residence': 65,
#     'date of birth': 66,
#     'date of death': 67,
#     'inception': 68,
#     'dissolved, abolished or demolished': 69,
#     'publication date': 70,
#     'start time': 71,
#     'end time': 72,
#     'point in time': 73,
#     'conflict': 74,
#     'characters': 75,
#     'lyrics by': 76,
#     'located on terrain feature': 77,
#     'participant': 78,
#     'influenced by': 79,
#     'location of formation': 80,
#     'parent organization': 81,
#     'notable work': 82,
#     'separated from': 83,
#     'narrative location': 84,
#     'work location': 85,
#     'applies to jurisdiction': 86,
#     'product or material produced': 87,
#     'unemployment rate': 88,
#     'territory claimed by': 89,
#     'participant of': 90,
#     'replaces': 91,
#     'replaced by': 92,
#     'capital of': 93,
#     'languages spoken, written or signed': 94,
#     'present in work': 95,
#     'sibling': 96
# }
INFINITY_NUMBER = 1e12
