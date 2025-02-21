from models import LargeLanguageModel
import pandas as pd

import os
import argparse
import random
import tiktoken
import pdb

import openai
import anthropic
import google.generativeai as genai
from prompts import get_solver_prompt
import datetime

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Output Directory')
	parser.add_argument('-data', type=str, default='chase_qa', help='Data name')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-prompt_type', type=str, default='zero-shot-basic', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-4o-mini', help='Which model to use')
	parser.add_argument('-peft_model', type=str, default='none', help='peft path')
	parser.add_argument('-max_tokens', type=int, default=2048, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.5, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	return parser

def solver(data, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	pred_ls = []

	tot_ip_tokens = 0
	tot_op_tokens = 0

	cnt = 0

	for i in range(len(data)):
		docs = data.loc[i]["Documents"]
		ques = data.loc[i]["Question"]
		ans = data.loc[i]["Answer"]

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Documents: \n\n" + docs + "\n\n")
			f.write("Question " + str(i+1) + ": " + ques + "\n")
			f.write("Ground Truth Answer: " + str(ans) + "\n\n")
		
		prompt, sys_prompt = get_solver_prompt(prompt_type, question=(docs, ques))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		tot_ip_tokens += ip_tokens
		tot_op_tokens += op_tokens

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Prediction: " + og_pred + "\n")
			f.write("------------------------------------------------------------------\n\n")

		pred_ls.append([i+1, docs, ques, ans, og_pred])
		cnt += 1

		pred_df = pd.DataFrame(pred_ls, columns = ['ID', 'Documents', 'Question', 'Answer', 'Prediction'])
		pred_df.to_csv(args.out_dir + "/predictions.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(data)), end = '\r', flush = True)

	print("Total input tokens: ", tot_ip_tokens)
	print("Total output tokens: ", tot_op_tokens)
	print("Answers generated: ", cnt)


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")
	
	_, sys_prompt = get_solver_prompt(args.prompt_type, question=("", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	data = pd.read_csv("data/" + args.data + ".tsv", sep='\t')
	solver(data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.data + "_" + args.prompt_type + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)