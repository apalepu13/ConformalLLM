import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import numpy as np

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def getModel(config):
    modelchoice = config['model']
    if modelchoice == 'llama' or modelchoice == 'alpaca':
        model = AutoModelForCausalLM.from_pretrained("<path_to_store_recovered_weights>")
        tokenizer = AutoTokenizer.from_pretrained("<path_to_store_recovered_weights>")
        model.train(False)
    elif modelchoice == 'alpaca':
        model = AutoModelForCausalLM.from_pretrained("<path_to_store_recovered_weights>")
        tokenizer = AutoTokenizer.from_pretrained("<path_to_store_recovered_weights>")
        model.train(False)
    elif modelchoice == 'stableLM':
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
        model.half().to(device)
        #prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

    return tokenizer, model

def getQuestions(s, config):
    if config['use_cot']:
        q_string = config['q_string_cot']
    else:
        q_string = config['q_string']
    q_strings = [q_string + q + "\n" for q in s['question']]
    q_strings = [q_strings[i] + "A: " + s['A'][i] + "\n" for i in range(len(s['A']))]
    q_strings = [q_strings[i] + "B: " + s['B'][i] + "\n" for i in range(len(s['B']))]
    q_strings = [q_strings[i] + "C: " + s['C'][i] + "\n" for i in range(len(s['C']))]
    q_strings = [q_strings[i] + "D: " + s['D'][i] + "\n" for i in range(len(s['D']))]
    q_strings = [q_strings[i] + "<|ASSISTANT|>Correct Answer: " for i in range(len(s['question']))]
    return q_strings

def parse_answer_choice(output):
    for char in output:
        if char == 'A' or char == 'B' or char == 'C' or char == 'D':
            return char

def self_consistency(dat, tokenizer, model, config, name="MedQA"):
    letters = ['A', 'B', 'C', 'D']
    model.train(False)
    with torch.no_grad():
        for i, s in enumerate(dat):
            if i > 10:
                break
            ans_choices = []
            q_string = getQuestions(s, config)[0]
            inputs = tokenizer(q_string, padding=False, truncation=False, return_tensors="pt").to(device)
            start_idx = len(inputs.input_ids[0])
            tokens = model.generate(
                **inputs,
                num_return_sequences=config['self_consistency_number'],
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()])
            )
            print("tokens",tokens)
            print("Input", q_string)
            for k, t in enumerate(tokens):
                output = tokenizer.decode(t[start_idx:], skip_special_tokens=True)
                choice = parse_answer_choice(output)
                print("output", k,output, choice)
                ans_choices.append(choice)
            ans_choices = np.array(ans_choices)
            choices, cts = np.unique(ans_choices, return_counts=True)
            print(letters[s['answer_idx'].item()], [str(choices[i]) + ": " + str(cts[i]) for i in range(len(choices))])




