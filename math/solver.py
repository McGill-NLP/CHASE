import pandas as pd

import os
import argparse
import json
import tiktoken
import random
import pdb

import openai
import anthropic
import google.generativeai as genai

from models import LargeLanguageModel
from prompts import get_solver_prompt

import datetime

def build_parser():
	parser = argparse.ArgumentParser(description='Solve')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=["\nQ:"], help='When to stop generation')

	parser.add_argument('-prompt_type', type=str, default='8-shot-cot', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic', 'o1'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-3.5-turbo', help='Which model to use')
	parser.add_argument('-peft_model', type=str, default='none', help='peft path')
	parser.add_argument('-max_tokens', type=int, default=1024, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.5, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-data', type=str, default='chase_math.tsv', help='Dataset name')

	return parser


def check_model_output(ques, num_answer, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	prompt, sys_prompt = get_solver_prompt(prompt_type, ques)

	og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

	ip_toks = len(tik_encoding.encode(prompt))
	op_toks = len(tik_encoding.encode(og_pred))

	pred = og_pred.lower().strip()
	if pred[-1] == ".":
		pred = pred[:-1]

	error = 0
	try:
		num_pred = float(pred.split("the answer is ")[1].replace("**", "").replace("<", "").replace(">", "").replace("}", "").replace("{", "").replace("\[", "").replace("\]", "").strip().split()[0].replace("$", "").replace(",", "").replace("%", ""))
	except:
		error = 1
		num_pred = -131.1133

	res = 0
	if abs(num_pred - num_answer) < 0.00001:
		res = 1

	return res, error, og_pred, num_pred, ip_toks, op_toks


def solve_and_evaluate(args, problems_data, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	predictions = []
	corr = 0
	err = 0
	checked = 0

	for i in range(len(problems_data)):
		ques = problems_data.loc[i]["question"]
		answer = problems_data.loc[i]["answer"]
		try:
			depth = problems_data.loc[i]["depth"]
		except:
			depth = 0
		try:
			if "####" in answer:
				num_answer = float(answer.split("####")[-1].strip())
			else:
				num_answer = float(answer.strip().split("\n")[-1])
		except:
			print("Error in parsing answer!!!")
			continue

		res, error, og_pred, num_pred, _, _ = check_model_output(ques, num_answer, model, prompt_type, max_tokens, temperature, stop, tik_encoding)

		checked += 1

		if error:
			err += 1

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Question " + str(i+1) + ":\n\n" + ques.encode('ascii', 'ignore').decode('ascii') + "\n\nGround-truth Answer:\n\n")
			f.write(answer.encode('ascii', 'ignore').decode('ascii') + "\n\nGround-truth Number: " + str(num_answer) + "\n\n")
			f.write("Depth: " + str(depth) + "\n\n")
			f.write("Prediction:\n\n")
			f.write(og_pred.encode('ascii', 'ignore').decode('ascii') + "\n\nPrediction Number: " + str(num_pred) + "\n\n")
			f.write("Error: " + str(error) + "\n\n")
			f.write("Result: " + str(res) + "\n\n")
			f.write("-------------------------------------------------------------------------------------\n\n")

		if res:
			corr += 1

		predictions.append([problems_data.loc[i]["question"], problems_data.loc[i]["answer"], depth, og_pred, res, error])

		pred_df = pd.DataFrame(predictions, columns = ['Question', 'Answer', 'Depth', 'Prediction', 'Result', 'Error'])

		try:
			pred_df.to_csv(args.out_dir + "/predictions.tsv", sep = '\t', index = None)
		except:
			predictions[-1][3] = "Error in saving!!!"

		print("Correct so far: " + str(corr) + " / " + str(checked) + " = " + str(corr/checked))

		print("Completed {} / {}...".format(i+1, len(problems_data)), end = '\r', flush = True)

	with open(args.out_dir + "/logs.txt", "a") as f:
		f.write("\n\n================================================================================================\n\n")
		f.write("Final Accuracy: " + str(corr) + " / " + str(checked) + " = " + str(corr/checked) + "\n\n")

	print("Final Accuracy: " + str(corr) + " / " + str(checked) + " = " + str(corr/checked))


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")
	
	_, sys_prompt = get_solver_prompt(args.prompt_type)

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model=args.peft_model, sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	data = pd.read_csv("data/" + args.data, sep = '\t')
	solve_and_evaluate(args, data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		if args.model_type == "peft":
			if args.peft_model == "none":
				args.run_name = args.data.replace(".tsv", "") + "_peft_base_" + args.prompt_type + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))
			else:
				args.run_name = args.data.replace(".tsv", "") + "_peft_finetuned_" + args.peft_model.split("/")[-1].strip()  + "_" + args.prompt_type + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))
		else:
			args.run_name = args.data.replace(".tsv", "") + "_" + args.model  + "_" + args.prompt_type + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)