import stanza
import json
import re
import copy
def stanford():
    # stanza.download('en') # download English model
    nlp = stanza.Pipeline('en',processors='tokenize,pos,depparse,lemma',tokenize_no_ssplit=True)

    content=read_json("../data/txt/test_triples.json")
    id = 0
    stanza_list = []

    for entry in content:
        text = entry["text"]
        triple_list = entry["triple_list"]
        entity_list = []
        for triple in triple_list:
            entity1, relation, entity2 = triple
            entity_string1 = "".join(entity1)
            entity_string2 = "".join(entity2)
            entity1 = entity_string1.split()
            entity2 = entity_string2.split()

            e1_s = re.split('[,/’-]', entity1[0])
            e1_e = re.split('[,/’-]', entity1[len(entity1)-1])
            e2_s = re.split('[,/’-]', entity2[0])
            e2_e = re.split('[,/’-]', entity2[len(entity2)-1])
            entity_list.append(e1_s[0])
            entity_list.append(e1_e[0])
            entity_list.append(e2_s[0])
            entity_list.append(e2_e[0])

        print(entity_list)
        doc = nlp(text)

        # 找到实体1和实体2在文本中的位置
        ent_num = -1
        entity_ids = [0] * 4
        for entity in entity_list:
            ent_num = ent_num + 1
            for sent in doc.sentences:
                for i, word in enumerate(sent.words):
                    if word.text == entity:
                        entity_ids[ent_num] = i
                        break

        ent_ids = copy.deepcopy(entity_ids)
        # 打印实体1和实体2在文本中的位置

        # 相似度匹配
        if(entity_ids[3] == 0):
            tokens = [word.text for sent in doc.sentences for word in sent.words]
            similarity_ent1_start = []
            similarity_ent1_end = []
            similarity_ent2_start = []
            similarity_ent2_end = []
            for i in range(len(tokens)):
                similarity_ent1_start.append(jaccard_similarity(entity_list[0],tokens[i]))
                similarity_ent1_end.append(jaccard_similarity(entity_list[1],tokens[i]))
                similarity_ent2_start.append(jaccard_similarity(entity_list[2], tokens[i]))
                similarity_ent2_end.append(jaccard_similarity(entity_list[3], tokens[i]))
                if similarity_ent1_start[i] > similarity_ent1_start[i-1]:
                    entity_ids[0] = i
                if similarity_ent1_end[i] > similarity_ent1_end[i-1]:
                    entity_ids[1] = i
                if similarity_ent2_start[i] > similarity_ent2_start[i-1]:
                    entity_ids[2] = i
                if similarity_ent2_end[i] > similarity_ent2_end[i-1]:
                    entity_ids[3] = i
            for i in range(len(ent_ids)):
                if ent_ids[i] != 0:
                    entity_ids[i] = ent_ids[i]
        print(entity_ids)
        start_entity1 = entity_ids[0]
        end_entity1 = entity_ids[1]
        start_entity2 = entity_ids[2]
        end_entity2 = entity_ids[3]

        tokens = [word.text for sent in doc.sentences for word in sent.words]
        pos_tags = [word.upos for sent in doc.sentences for word in sent.words]
        head_indexes = [word.head for sent in doc.sentences for word in sent.words]
        deprels = [word.deprel for sent in doc.sentences for word in sent.words]

        json_data = {
            "id": id,
            "token": tokens,
            "stanford_pos": pos_tags,
            "stanford_head": head_indexes,
            "stanford_deprel": deprels,
            "first_start": start_entity1,
            "first_end":end_entity1,
            "second_start": start_entity2,
            "second_end": end_entity2,
            "relation": relation
        }
        id = id + 1
        stanza_list.append(json_data)
        print(json_data)
        print("=" * 30)
    return stanza_list

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    return content

def write_json(stanza_list):
        with open("../data/conllx/test.json", 'w', encoding='utf-8') as file:
            json.dump(stanza_list, file, indent=1, ensure_ascii=False)


def main():
    stanza_list = stanford()
    write_json(stanza_list)

def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union
    return similarity

if __name__ == "__main__":
    main()
