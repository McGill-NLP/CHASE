import pandas as pd

import os
import argparse
import json
import tiktoken
import pdb

import openai
import anthropic
import google.generativeai as genai

from models import LargeLanguageModel
from prompts import get_evaluator_prompt

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-exp_type', type=str, default='model', help='Exp type')
	parser.add_argument('-run_name', type=str, default='chase_qa_llama', help='run name')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Output Directory')
	parser.add_argument('-data', type=str, default='chase_qa', help='Output Directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-prompt_type', type=str, default='zero-shot-basic', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-4o', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=1024, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.5, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	return parser

def evaluator(pred_data, full_data, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0
	score = 0.0

	inc = 0.0
	irr = 0.0
	adv = 0.0

	for i in range(len(pred_data)):
		id1 = pred_data.loc[i]["ID"]
		ques = pred_data.loc[i]["Question"]
		ans = pred_data.loc[i]["Answer"]
		pred = pred_data.loc[i]["Prediction"]

		idx = full_data.index[(full_data["Answer"] == ans) & (full_data["Question"] == ques)].tolist()
		assert len(idx) == 1
		idx = idx[0]

		try:
			adv_answer = json.loads(full_data.loc[idx]['Adv_Answer'])
		except:
			adv_answer = []
		
		adv_ans_str = ""
		for adv_ans in adv_answer:
			adv_ans_str = adv_ans_str + adv_ans + "\n"
		adv_ans_str = adv_ans_str.strip()

		with open(args.out_dir + "/eval_logs.txt", "a") as f:
			f.write("Question " + str(id1) + ": " + ques + "\n\n")
			f.write("Ground Truth Answer: " + str(ans) + "\n\n")
			f.write("Adversarial Answer: " + str(adv_ans_str) + "\n\n")
			f.write("Prediction: " + str(pred) + "\n\n")
		
		prompt, sys_prompt = get_evaluator_prompt(prompt_type, question=(ques, ans, pred, adv_ans_str))
		
		if pred.strip() == "error":
			og_pred = "False. Error in prediction."
		else:
			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		tot_ip_tokens += ip_tokens
		tot_op_tokens += op_tokens

		with open(args.out_dir + "/eval_logs.txt", "a") as f:
			f.write("Result: " + og_pred + "\n")
			f.write("------------------------------------------------------------------\n\n")

		res = og_pred.strip().split()[0]

		if "score" in args.prompt_type:
			if "." in res or "," in res:
				res = res[:-1]
			corr = int(res)
		elif "step-by-step" in args.prompt_type:
			corr = 0
			if "FINAL VERDICT: Correct" in og_pred:
				corr = 1
			if "correct" in og_pred.strip().split("\n")[-1].lower():
				corr = 1
		else:
			corr = 0
			if "true" in res.lower():
				corr = 1

		score += corr

		pred_ls.append([i+1, ques, ans, pred, corr, og_pred])
		cnt += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Question', 'Answer', 'Prediction', 'Result', 'Explanation'])
		pred_df.to_csv(args.out_dir + "/result.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(pred_data)), end = '\r', flush = True)

	inc_score = 0.0
	irr_score = 0.0
	adv_score = 0.0
	if cnt-score > 0:
		inc_score = inc/(cnt-score)
		irr_score = irr/(cnt-score)
		adv_score = adv/(cnt-score)

	with open(args.out_dir + "/eval_logs.txt", "a") as f:
		f.write("Accuracy: " + str(score/cnt))
		f.write("% Incorrect: " + str(inc_score))
		f.write("% Irrelevant: " + str(irr_score))
		f.write("% Adversarial: " + str(adv_score))

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Questions evaluated: ", cnt)
	print()
	print("Accuracy: ", score/cnt)


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")
	
	_, sys_prompt = get_evaluator_prompt(args.prompt_type, question=("", "", "", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	predictions_data = pd.read_csv(args.out_dir + "/predictions.tsv", sep='\t')
	full_data = pd.read_csv("data/" + args.data + ".tsv", sep='\t')

	if args.exp_type == "model":
		evaluator(predictions_data, full_data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)