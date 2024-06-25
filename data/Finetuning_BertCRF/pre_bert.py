from data.Finetuning_BertCRF.Bert_Feature import GetBertFeature
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bert = GetBertFeature(device,50,50)

# sents = [["icotinib", "ren", "et", "al.", "[", "]", "ph", "i", ";", "single-arm", "n", "=", "7", ":", "chinese", ";", "previously", "treat", ";", "exon", "19", "deletion", "(", "n", "=", "3", ")", ";", "l858r", "(", "n", "=", "4", ")", "icotinib", "(", "varied", "dose", "and", "schedule", ")", "pfs", ":", "141", "day", "(", "4.6", "month", ")", "sun", "et", "al.", "[", "]", "ph", "iii", ";", "randomized", "comparison", "with", "gefitinib", "(", "icogen", ")", "n", "=", "27", ":", "chinese", ";", "previously", "treat", ";", "egfr", "mutation", "icotinib", "125", "mg", "three", "times/day", "pfs", ":", "198", "day", "(", "6.5", "month", ")"]]
# lables = [["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O","O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O","O", "O", "O", "O", "O", "O"]]
sents = [["icotinib", "ren", "et"]]
lables = [["O", "O", "O"]]
bert_emission = bert.get_bert_feature(sents,lables,device,50,50)

print(bert_emission[0][:6])