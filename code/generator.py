import pandas as pd

import os
import argparse
import random
import json
import tiktoken
import re
import pdb

import openai
import anthropic
import google.generativeai as genai

from models import LargeLanguageModel
from prompts import get_generator_prompt
from utils import *
from executions import *
from verifications import *
from context_utils import *

import datetime

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-exp_type', type=str, default='helper_functions', help='Exp type')
	parser.add_argument('-prompt_type', type=str, default='helper_functions', help='prompt type')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-3.5-turbo', help='Which model to use')
	parser.add_argument('-supp_model_type', type=str, default='vllm', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-supp_model', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=8000, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=1.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-domain', type=str, default='data_preprocessing', help='Exp type')
	parser.add_argument('-num_iters', type=int, default=3, help='number of iterations')

	parser.add_argument('-extra_fns', type=int, default=10, help='number of iterations')

	parser.add_argument('-continue_problems_name', type=str, default='default', help='Continuing Problems name')
	parser.add_argument('-continue_test_name', type=str, default='default', help='Continuing Test name')
	parser.add_argument('-problems_name', type=str, default='gpt-4o-mini-problems', help='Problems name')
	parser.add_argument('-tested_name', type=str, default='gpt-4o-mini-tests', help='Tested data name')
	parser.add_argument('-verified_name', type=str, default='gpt-4o-mini-verified', help='Verified data name')
	parser.add_argument('-verified_filename', type=str, default='verified_problems.tsv', help='Verified filename')

	return parser


def helper_generation(args, model, prompt_type, domain="data_preprocessing", num_iters=5, max_tokens=8000, temperature=1.0, stop=[], tik_encoding=None):
	if os.path.exists("helper_functions/generated/" + domain + ".json"):
		with open("helper_functions/generated/" + domain + ".json", "r") as f:
			functions = json.load(f)
		print("Loaded {} functions...".format(len(functions)))
	else:
		functions = []
	for i in range(num_iters):
		corr_fns = 0
		if i < 15 and len(functions) < 50:
			with open("helper_functions/annotated/" + domain + ".txt", "r") as f:
				seed_functions_txt = f.read()
		else:
			_, seed_functions_txt = obtain_seed_functions(domain, 5)

		prompt, sys_prompt = get_generator_prompt(prompt_type, params=(domain, seed_functions_txt))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("===============================================================================================\n\n")
			f.write("Model output:\n\n" + og_pred + "\n\n")
			f.write("---------------------------------------------------------\n\n")

		try:
			cur_functions = get_functions(og_pred)
		except Exception as e:
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("============= Function Parsing Error =============\n\n")
				f.write(og_pred + "\n\n")
				f.write("Error:\n" + str(e) + "\n\n")
				f.write("==================================================\n\n")
			continue
		
		for cur_function in cur_functions:
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Trying function " + cur_function["function_name"] + "...\n\n")
			sim_check = False
			for func in functions:
				name1 = cur_function["function_name"].split("_")
				filename1 = cur_function["file_name"].replace(".py","").split("_")
				name2 = func["function_name"].split("_")
				filename2 = func["file_name"].replace(".py","").split("_")
				if jaccard_similarity(set(name1 + filename1), set(name2 + filename2)) > 30:
					obj1 = set("\n".join(cur_function["objectives"]).split())
					obj2 = set("\n".join(func["objectives"]).split())
					if jaccard_similarity(obj1, obj2) > 20:
						sim_check = True
						with open(args.out_dir + "/logs.txt", "a") as f:
							f.write("------------- SIMILAR FUNCTION FOUND -------------\n\n")
							f.write(obtain_functions_text([cur_function]).strip() + "\n\n")
							f.write("--------------------------------------------------\n\n")
						break
			if not sim_check:
				test_code = get_function_exec_test(model, cur_function)
				exec_res, exec_op = helper_exec_check(cur_function, test_code)
				if exec_res:
					corr_fns += 1
					functions.append(cur_function)

					with open(args.out_dir + "/logs.txt", "a") as f:
						f.write(obtain_functions_text([cur_function]).strip() + "\n\n")
						f.write("----------------------------------------------------------------------------\n\n")
				else:
					with open(args.out_dir + "/logs.txt", "a") as f:
						f.write("------ Helper Function Execution Failed ------\n\n")
						f.write(obtain_functions_text([cur_function]).strip() + "\n\n")
						f.write("Error:\n" + exec_op + "\n\n")
						f.write("----------------------------------------------\n\n")

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("======================================================================================================\n\n")

		with open("helper_functions/generated/" + domain + "_new.json", 'w') as json_file:
			json.dump(functions, json_file, indent=4)

		print("{} functions added... {} in total.".format(corr_fns, len(functions)))

		print("Completed {} / {}...".format(i+1, num_iters), end = '\r', flush = True)


def problem_generation(args, problems_data, model, prompt_type, domain="data_preprocessing", num_iters=20, max_tokens=8000, temperature=1.0, stop=[], tik_encoding=None):
	problems = []
	if problems_data is not None:
		for j in range(len(problems_data)):
			problems.append([json.loads(problems_data.loc[j]["Prompt Functions"]), json.loads(problems_data.loc[j]["Relevant Functions"]), problems_data.loc[j]["Problem"], json.loads(problems_data.loc[j]["Answer"])])
		print("Loaded {} problems...".format(len(problems)))
		
	for i in range(num_iters):
		funcs, functions_txt = obtain_seed_functions(domain, 10)

		prompt, sys_prompt = get_generator_prompt(prompt_type, params=(domain, functions_txt.strip()))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Input Tokens: " + str(ip_tokens) + "\n")
			f.write("Output Tokens: " + str(op_tokens) + "\n\n")
			f.write("Model output:\n\n" + og_pred + "\n\n")
			f.write("---------------------------------------------------------\n\n")

		try:
			prelim_prob_stmt, function_name, file_name, import_lines, function_def, rel_funcs = process_output(og_pred)
			if not para_return_check(function_def, prelim_prob_stmt):
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("---------------------- Para Return Check Failed -----------------------------\n\n")
					f.write("==========================================================================================\n\n")
				continue
		except:
			print("Failed to Parse...")
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("---------------------- Failed to Parse -----------------------------\n\n")
				f.write("==========================================================================================\n\n")
			continue

		relevant_functions = []

		rel_funcs_flag = False

		for rel_func in rel_funcs:
			func_names = rel_func[0].split(",")
			func_file = rel_func[1] + ".py"
			for func_name in func_names:
				found_flag = False
				for func in funcs:
					if func_name.strip() == func["function_name"] and func_file == func["file_name"]:
						relevant_functions.append(func)
						found_flag = True
						break
				if not found_flag:
					print("Not Found: ", rel_func)
					rel_funcs_flag = True
					with open(args.out_dir + "/logs.txt", "a") as f:
						f.write("---------------------- Relevant Function Not Found -----------------------------\n\n")
						f.write(str(rel_func) + "\n\n")
						f.write("==========================================================================================\n\n")
					continue

		if rel_funcs_flag:
			continue

		ans_code = {
			"function_name": function_name,
			"file_name": file_name,
			"import_lines": import_lines,
			"function_def": function_def
		}

		prelim_prob_stmt = remove_func_mentions(prelim_prob_stmt, relevant_functions)

		test_code = get_function_exec_test(model, ans_code)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Test Code:\n\n" + test_code + "\n\n")

		exec_res, exec_op = problem_exec_check(relevant_functions, ans_code, test_code)
		if not exec_res:
			print("Generated complex function did not execute correctly...")
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("---------------------- Execution Failed -----------------------------\n\n")
				f.write("==========================================================================================\n\n")
			continue

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("---------------------- Execution Passed -----------------------------\n\n")
		
		problems.append([funcs, relevant_functions, prelim_prob_stmt, ans_code])

		print("{} successful problems created so far...".format(len(problems)))

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Problem Statement " + str(len(problems)) + ":\n\n" + prelim_prob_stmt + "\n\nAnswer Code:\n\n")
			f.write(obtain_main_fn_txt(ans_code).strip() + "\n\n")
			f.write("========================================================================================\n\n")

		pred_df = pd.DataFrame(problems, columns = ['Prompt Functions', 'Relevant Functions', 'Problem', 'Answer'])
		pred_df['Prompt Functions'] = pred_df['Prompt Functions'].apply(json.dumps)
		pred_df['Relevant Functions'] = pred_df['Relevant Functions'].apply(json.dumps)
		pred_df['Answer'] = pred_df['Answer'].apply(json.dumps)
		pred_df.to_csv(args.out_dir + "/problems.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, num_iters), end = '\r', flush = True)

	print("Successfully generated {} problems...".format(len(problems)))


def test_generation(args, already_tested_data, problems_data, model, prompt_type, num_iters, max_tokens, temperature, stop, tik_encoding):
	data = []
	if already_tested_data is not None:
		for j in range(len(already_tested_data)):
			data.append([already_tested_data.loc[j]["Prompt Functions"], already_tested_data.loc[j]["Relevant Functions"], already_tested_data.loc[j]["Problem"], already_tested_data.loc[j]["Answer"], already_tested_data.loc[j]["Test"]])
		print("Loaded {} problems...".format(len(data)))

	all_covered = len(data)

	for i in range(len(problems_data)):
		prob_found = False
		# pdb.set_trace()
		for k in range(len(data)):
			if problems_data.loc[i]["Problem"] == data[k][2]:
				prob_found = True
				break
		if prob_found:
			all_covered -= 1
			continue
		if all_covered > 0:
			continue
		print("Starting from problem {}...".format(i+1))
		for j in range(num_iters):
			rel_functions = json.loads(problems_data.loc[i]["Relevant Functions"])
			codebase = obtain_functions_text(rel_functions).strip()
			ans_code = json.loads(problems_data.loc[i]["Answer"])

			main_fn_txt = obtain_main_fn_txt(ans_code)

			prompt, sys_prompt = get_generator_prompt(prompt_type, params=(codebase, main_fn_txt))

			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

			ip_tokens = len(tik_encoding.encode(prompt))
			op_tokens = len(tik_encoding.encode(og_pred))

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Input Tokens: " + str(ip_tokens) + "\n")
				f.write("Output Tokens: " + str(op_tokens) + "\n\n")
				f.write("Full prediction:\n\n" + og_pred + "\n\n")

			try:
				if "```Python" in og_pred:
					test_code = og_pred.split("```Python")[1].split("```")[0]
				elif "```python" in og_pred:
					test_code = og_pred.split("```python")[1].split("```")[0]
				test_code = correct_import_line(test_code)
				test_code = re.sub(r'""".*?"""', '', test_code, flags=re.DOTALL)

				assert_flag = assert_check(test_code)
				
				if not assert_flag:
					test_code = test_code.replace("# assert", "assert")

					assert_flag = assert_check(test_code)

					if not assert_flag:
						with open(args.out_dir + "/logs.txt", "a") as f:
							f.write("No assert statement in test code...\n\n")
						raise Exception("No assert statement in test code...")
			except:
				print("Error in test code parsing...")
				print("Trial {}: Failure in testing!".format(j+1))
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Error in test code parsing...\n\n")
					f.write("Trial {}: Failure in testing!\n\n".format(j+1))
					f.write("-------------------------------------------------------------------------------\n\n")
				continue
			test_code = test_code + "\n\nprint('All-Pass')"

			test_ver = verify_test_code(test_code, ans_code['function_name'])
			if not test_ver:
				print("Trial {}: Failure in testing!".format(j+1))
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Test code verification failed...\n\n")
					f.write("Trial {}: Failure in testing!\n\n".format(j+1))
					f.write("-------------------------------------------------------------------------------\n\n")
				continue

			rel_func_presence = check_rel_func_test(test_code, rel_functions)
			if rel_func_presence:
				print("Trial {}: Failure in testing!".format(j+1))
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Relevant functions are called in test code...\n\n")
					f.write("Trial {}: Failure in testing!\n\n".format(j+1))
					f.write("-------------------------------------------------------------------------------\n\n")
				continue

			exec_res1, exec_op1 = execution(rel_functions, ans_code, test_code)
			exec_res2, exec_op2 = execution(rel_functions, ans_code, test_code)

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("===================================================================================\n\n")
				f.write("Problem Statement " + str(i+1) + ":\n\n" + problems_data.loc[i]["Problem"] + "\n\nAnswer Code:\n\n")
				f.write(obtain_main_fn_txt(ans_code).strip() + "\n\n")
				f.write("-----------------------TEST CODE--------------------------\n\n")
				f.write(test_code + "\n\n")
				f.write("-----------------------Execution--------------------------\n\n")
				f.write("Execution 1 Result: " + str(exec_res1) + "\n\n")
				f.write("Execution 1 Output: " + exec_op1 + "\n\n")
				f.write("Execution 2 Result: " + str(exec_res2) + "\n\n")
				f.write("Execution 2 Output: " + exec_op2 + "\n\n")
				f.write("----------------------------------------------------------\n\n")

			if exec_res1 and exec_res2:
				data.append([problems_data.loc[i]["Prompt Functions"], problems_data.loc[i]["Relevant Functions"], problems_data.loc[i]["Problem"], problems_data.loc[i]["Answer"], test_code])

				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("TEST EXECUTED SUCCESSFULLY\n\n")

				pred_df = pd.DataFrame(data, columns = ['Prompt Functions', 'Relevant Functions', 'Problem', 'Answer', 'Test'])

				pred_df.to_csv(args.out_dir + "/tested_problems.tsv", sep = '\t', index = None)

				print("Trial {}: Success! Total Successful Problems: {}".format(j+1, len(data)))
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Trial {}: Success! Total Successful Problems: {}\n\n".format(j+1, len(data)))
					f.write("===================================================================================\n\n")
				break
			else:
				print("Trial {}: Failure in testing!".format(j+1))
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Trial {}: Failure in testing!\n\n".format(j+1))
					f.write("-------------------------------------------------------------------------------\n\n")

		print("Completed {} / {}...".format(i+1, len(problems_data)), end = '\r', flush = True)

	print("Successfully generated {} problems...".format(len(data)))


def verified_generation(args, problems_data, model, prompt_type, domain, num_iters, max_tokens, temperature, stop, tik_encoding):
	data = []

	for i in range(len(problems_data)):
		rel_functions = json.loads(problems_data.loc[i]["Relevant Functions"])
		codebase = obtain_functions_text(rel_functions).strip()
		
		problem_statement = problems_data.loc[i]["Problem"]
		ans_code = json.loads(problems_data.loc[i]["Answer"])
		test_code = problems_data.loc[i]["Test"]

		prompt, sys_prompt = get_generator_prompt(prompt_type, (domain, codebase, problem_statement))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		ip_tokens = len(tik_encoding.encode(prompt))
		op_tokens = len(tik_encoding.encode(og_pred))

		if "```Python" in og_pred:
			pred_code = og_pred.split("```Python")[1].split("```")[0].strip()
		elif "```python" in og_pred:
			pred_code = og_pred.split("```python")[1].split("```")[0].strip()

		pred_code = pred_code.replace(".py import", " import")

		res, _ = prediction_check(rel_functions, pred_code, ans_code, test_code)

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Input Tokens: " + str(ip_tokens) + "\n")
			f.write("Output Tokens: " + str(op_tokens) + "\n\n")
			f.write("Problem Statement " + str(i+1) + ":\n\n" + problem_statement + "\n\nAnswer Code:\n\n")
			f.write(obtain_main_fn_txt(ans_code).strip() + "\n\n")
			f.write("Predicted Code:\n\n")
			f.write(pred_code + "\n\n")
			f.write("Execution Result: " + str(res) + "\n\n")
		
		succ = res

		if not res:
			ctr = 0
			while(ctr < num_iters):
				pred_fn_def = "def " + pred_code.split("def ")[1].strip()
				prompt, sys_prompt = get_generator_prompt("modify_problems", (problem_statement, pred_fn_def, ans_code["function_def"]))

				modified_problem_statement = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

				ip_tokens = len(tik_encoding.encode(prompt))
				op_tokens = len(tik_encoding.encode(modified_problem_statement))

				modified_problem_statement = remove_func_mentions(modified_problem_statement, rel_functions)

				prompt, sys_prompt = get_generator_prompt(prompt_type, (domain, codebase, modified_problem_statement))

				new_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

				if "```Python" in new_pred:
					new_pred = new_pred.split("```Python")[1].split("```")[0].strip()
				elif "```python" in new_pred:
					new_pred = new_pred.split("```python")[1].split("```")[0].strip()
					
				new_pred = new_pred.replace(".py import", " import")

				new_res, _ = prediction_check(rel_functions, new_pred, ans_code, test_code)

				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Trial " + str(ctr+1) + ":\n\n")
					f.write("Input Tokens: " + str(ip_tokens) + "\n")
					f.write("Output Tokens: " + str(op_tokens) + "\n\n")
					f.write("Modified Problem Statement " + str(i+1) + ":\n\n" + modified_problem_statement + "\n\n")
					f.write("New Predicted Code:\n\n")
					f.write(new_pred + "\n\n")
					f.write("New Execution Result: " + str(new_res) + "\n\n")

				if not new_res:
					with open(args.out_dir + "/logs.txt", "a") as f:
						f.write("Trial RESULT: FAILED\n\n")
						f.write("----------------------------------------------------------------------------------------\n\n")
					ctr += 1
				else:
					succ = 1
					problem_statement = modified_problem_statement
					break

		if not succ:
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("FINAL RESULT: FAILED\n\n")
				f.write("========================================================================================\n\n")
			print("Completed {} / {}...".format(i+1, len(problems_data)), end = '\r', flush = True)
			continue
		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("FINAL RESULT: PASSED\n\n")
			f.write("========================================================================================\n\n")

		data.append([problems_data.loc[i]["Prompt Functions"], problems_data.loc[i]["Relevant Functions"], problem_statement, problems_data.loc[i]["Answer"], test_code])

		pred_df = pd.DataFrame(data, columns = ['Prompt Functions', 'Relevant Functions', 'Problem', 'Answer', 'Test'])

		pred_df.to_csv(args.out_dir + "/verified_problems.tsv", sep = '\t', index = None)

		print("Success! Total Successful Problems: {}".format(len(data)))

		print("Completed {} / {}...".format(i+1, len(problems_data)), end = '\r', flush = True)

	print("Successfully generated {} problems...".format(len(data)))


def final_data_generation(args, verified_data, tik_encoding):
	data = []
	maxlen = 0
	avglen = 0

	for i in range(len(verified_data)):
		prompt_functions = json.loads(verified_data.loc[i]["Prompt Functions"])

		files, context = sample_context(prompt_functions, args.domain, args.extra_fns)

		if len(tik_encoding.encode(context)) > maxlen:
			maxlen = len(tik_encoding.encode(context))
		avglen += len(tik_encoding.encode(context))
		
		data.append([verified_data.loc[i]["Prompt Functions"], verified_data.loc[i]["Relevant Functions"], files, context, verified_data.loc[i]["Problem"], verified_data.loc[i]["Answer"], verified_data.loc[i]["Test"]])

		print("Completed {} / {}...".format(i+1, len(verified_data)), end = '\r', flush = True)
	
	pred_df = pd.DataFrame(data, columns = ['Prompt Functions', 'Relevant Functions', 'Codebase', 'Context', 'Problem', 'Answer', 'Test'])
	pred_df['Codebase'] = pred_df['Codebase'].apply(json.dumps)

	pred_df.to_csv(args.out_dir + "/final_data.tsv", sep = '\t', index = None)

	print("Max Context Length: ", maxlen)
	print("Avg Context Length: ", avglen/len(pred_df))

	print("Successfully generated {} problems...".format(len(data)))


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4o")
	
	_, sys_prompt = get_generator_prompt(args.prompt_type, params=("", "", "", "", "", "", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	supp_model = LargeLanguageModel(model_type=args.supp_model_type, model=args.supp_model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	if args.exp_type == "helper_functions":
		helper_generation(args, model, args.prompt_type, args.domain, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "problem_statement":
		if args.continue_problems_name != "default":
			args.out_dir = os.path.join(args.out_dir_name, args.continue_problems_name)
			problems_data = pd.read_csv(args.out_dir + "/problems.tsv", sep = '\t')
		else:
			problems_data = None
		problem_generation(args, problems_data, model, args.prompt_type, args.domain, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "test":
		if args.continue_test_name != "default":
			args.out_dir = os.path.join(args.out_dir_name, args.continue_test_name)
			already_tested_data = pd.read_csv(args.out_dir + "/tested_problems.tsv", sep = '\t')
		else:
			already_tested_data = None
		problems_data = pd.read_csv(args.out_dir_name + args.problems_name + "/problems.tsv", sep = '\t')
		test_generation(args, already_tested_data, problems_data, model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "verify_problems":
		tested_data = pd.read_csv(args.out_dir_name + args.tested_name + "/tested_problems.tsv", sep = '\t')
		verified_generation(args, tested_data, model, args.prompt_type, args.domain, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "final_data":
		verified_data = pd.read_csv(args.out_dir_name + args.verified_name + "/" + args.verified_filename, sep = '\t')
		final_data_generation(args, verified_data, tik_encoding)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.domain + "_" + args.exp_type + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)