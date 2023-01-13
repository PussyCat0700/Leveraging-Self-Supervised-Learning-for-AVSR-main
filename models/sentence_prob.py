import torch
#torch.cuda.set_device(7)
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

gpuAvailable = torch.cuda.is_available()
device = torch.device("cuda" if gpuAvailable else "cpu")
tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")   #这个有什么说法吗
model.to(device)

#inputs = tokenizer("I am happy", return_tensors="pt")
#outputs = model(**inputs, labels=inputs["input_ids"]) #(1,3,267735) # tokens= tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) 感觉这句相当于验证
#outputs = model(**inputs, labels=None)


def p(sentence):
    # input: "I am happy" output p(happy| I am)
    inputs = tokenizer(sentence, return_tensors="pt")    #inputs["input_ids"][0]  tensor([68, 2271, 3463])
    inputs.to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    losses=outputs.losses  # [4.5008, 8.4680] Negative log likelihood   [(len-1)*bsz] Negative log likelihood.
    prob = torch.exp(-losses)  #tensor([[0.0111, 0.0002]]   #prob= torch.prod(prob)  #2.3320e-06
    prob = prob.tolist()[0][-1]   #[[0.011100328527390957, 0.00021007754548918456]]
    return prob

# print(p("I am happy"))
# print(p("I am not happy"))
# print(p("She am happy"))
# print(p("She am not happy"))
# print(p("I am I am I am"))
# print(p("Today is Sunday"))
# print(p("a loud laugh followed at chunkeys expenses"))   #5.6475e-06
# print(p("but no gost"))