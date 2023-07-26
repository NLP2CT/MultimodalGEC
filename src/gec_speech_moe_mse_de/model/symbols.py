symbols_bert = {
    "BOTgt": "[unused1]",
    "EOTgt": "[unused2]",
    "PAD": "[PAD]",
    "BOSrc": "[CLS]",
    "EOSrc": "[SEP]",
}


symbols_t5 = {
    "BOTgt": "<pad>",
    "EOTgt": "</s>",
    "PAD": "<pad>",
    "BOSrc": "<s>",
    "EOSrc": "</s>",
}


symbols_bart = {
    "BOTgt": "<s>",
    "EOTgt": "</s>",
    "PAD": "<pad>",
    "BOSrc": "<s>",
    "EOSrc": "</s>",
}


# symbols_bart = {
#     "BOTgt": "<0x00>",
#     "EOTgt": "</s>",
#     "PAD": "<pad>",
#     "BOSrc": "<s>",
#     "EOSrc": "</s>",
# }


symbols_longformer = {
    "BOTgt": "[unused1]",
    "EOTgt": "[unused2]",
    "PAD": "[PAD]",
    "BOSrc": "[CLS]",
    "EOSrc": "[SEP]",
}

symbols_dict = dict()
symbols_dict['symbols_bert'] = symbols_bert
symbols_dict['symbols_bart'] = symbols_bart
symbols_dict['symbols_t5'] = symbols_t5
symbols_dict['symbols_longformer'] = symbols_longformer
