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
from prompts import get_solver_prompt
from utils import *
from executions import *

import datetime

def build_parser():
	parser = argparse.ArgumentParser(description='Solve')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')

	parser.add_argument('-prompt_type', type=str, default='basic', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-3.5-turbo', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=4096, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=1.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-data', type=str, default='chase_code_dp.tsv', help='Dataset name')

	return parser


def solve_and_evaluate(args, problems_data, model, prompt_type, max_tokens, temperature, stop, tik_encoding):
	predictions = []
	corr = 0
	err = 0

	for i in range(len(problems_data)):
		rel_funcs = json.loads(problems_data.loc[i]["Relevant Functions"])
		raw_codebase = json.loads(problems_data.loc[i]["Codebase"])
		context = problems_data.loc[i]["Context"]
		problem_statement = problems_data.loc[i]["Problem"]
		ans_code = json.loads(problems_data.loc[i]["Answer"])
		test_code = problems_data.loc[i]["Test"]

		prompt, sys_prompt = get_solver_prompt(prompt_type, (context, problem_statement))

		if prompt.count("The name of the function you create should be") > 1:
			lines = prompt.split("\n")
			new_prompt = ""
			cnt = 0
			for line in lines:
				if "The name of the function you create should be" in line:
					cnt += 1
					if cnt == 1:
						new_prompt += line + "\n"
				else:
					new_prompt += line + "\n"
			prompt = new_prompt

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		pred_code = remove_python_wrapper(og_pred)

		called_funcs = get_called_functions(pred_code)

		ans_code_text = "\n".join(obtain_main_fn_txt(ans_code).split("\n")[1:]).strip()

		sanity, _ = evaluate_prediction(raw_codebase, ans_code_text, ans_code, test_code)

		if not sanity:
			print("Sanity failed: Even the answer code is not working.")
			err += 1
			continue

		res, exec_op = evaluate_prediction(raw_codebase, pred_code, ans_code, test_code)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Problem Statement " + str(i+1) + ":\n\n" + problem_statement + "\n\nAnswer Code:\n\n")
			f.write(obtain_main_fn_txt(ans_code).strip() + "\n\n")
			f.write("----------------------------------------\n\n")
			f.write("Predicted Code:\n\n")
			f.write(pred_code + "\n\n")
			f.write("----------------------------------------\n\n")
			f.write("Relevant Functions:\n\n")
			for rel_func in rel_funcs:
				f.write(rel_func["function_def"] + "\n\n")
			f.write("----------------------------------------\n\n")
			f.write("\nCalled Functions:\n\n")
			for file_name in list(raw_codebase.keys()):
				file_funcs = raw_codebase[file_name]
				for ff in file_funcs:
					if ff["function_name"] in called_funcs:
						f.write(ff["function_def"] + "\n\n")
			f.write("----------------------------------------\n\n")
			f.write("Test Code:\n\n")
			f.write(test_code + "\n\n")
			f.write("----------------------------------------\n\n")
			f.write("Execution Result: " + str(res) + "\n\n")
			f.write("Execution Output:\n\n" + exec_op + "\n\n")
			f.write("---------------------------------------------------------------------------------------------\n\n")

		if res:
			corr += 1

		predictions.append([problems_data.loc[i]["Prompt Functions"], problems_data.loc[i]["Relevant Functions"], problems_data.loc[i]["Codebase"], context, problem_statement, problems_data.loc[i]["Answer"], pred_code, test_code, res])

		pred_df = pd.DataFrame(predictions, columns = ['Prompt Functions', 'Relevant Functions', 'Codebase', 'Context', 'Problem', 'Answer', 'Prediction', 'Test', 'Result'])

		pred_df.to_csv(args.out_dir + "/predictions.tsv", sep = '\t', index = None)

		print("Correct so far: " + str(corr) + " / " + str(i+1) + " = " + str(corr/(i+1)))

		print("Completed {} / {}...".format(i+1, len(problems_data)), end = '\r', flush = True)

	with open(args.out_dir + "/logs.txt", "a") as f:
		f.write("\n\n================================================================================================\n\n")
		f.write("Final Accuracy: " + str(corr) + " / " + str(len(problems_data)-err) + " = " + str(corr/(len(problems_data)-err)) + "\n\n")

	print("Final Accuracy: " + str(corr) + " / " + str(len(problems_data)-err) + " = " + str(corr/(len(problems_data)-err)))


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")
	
	_, sys_prompt = get_solver_prompt(args.prompt_type, params=("", "", "", "", "", "", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	data = pd.read_csv("data/" + args.data, sep = '\t')
	solve_and_evaluate(args, data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.data.replace(".tsv", "") + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)