import pandas as pd

import os
import argparse
import random
import json
import tiktoken
import pdb

import openai
import anthropic
import google.generativeai as genai

from models import LargeLanguageModel
from prompts import get_generator_prompt, get_verification_prompt

from utils import process_output, process_grammar_corrected, nums_check, find_longest_repeated_phrase, process_naive
from solver import check_model_output

import datetime
import time

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-exp_type', type=str, default='problem_extend', help='Exp type')
	parser.add_argument('-prompt_type', type=str, default='problem_extend', help='prompt type')
	parser.add_argument('-indi_verify', type=int, default=0, help='Whether to individually verify new problems by themselves?')
	parser.add_argument('-model_type', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='gpt-4o-mini', help='Which model to use')
	parser.add_argument('-ver_model_type_1', type=str, default='chat', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-ver_model_1', type=str, default='gpt-4o-mini', help='Which model to use')
	parser.add_argument('-ver_model_type_2', type=str, default='gemini', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-ver_model_2', type=str, default='gemini-1.5-flash', help='Which model to use')
	parser.add_argument('-overseer', type=int, default=0, help='Whether to use overseer model')
	parser.add_argument('-overseer_model_type', type=str, default='vllm', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-overseer_model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Which model to use')
	parser.add_argument('-grammar', type=int, default=0, help='Whether to use grammar model')
	parser.add_argument('-grammar_model_type', type=str, default='gemini', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-grammar_model', type=str, default='gemini-1.5-flash', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=4096, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=1.0, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')

	parser.add_argument('-num_iters', type=int, default=10, help='number of iterations')
	parser.add_argument('-max_depth', type=int, default=5, help='number of iterations')
	parser.add_argument('-min_depth', type=int, default=2, help='number of iterations')

	parser.add_argument('-seed_name', type=str, default='seeds', help='seed data name')
	parser.add_argument('-continue_name', type=str, default='default', help='Continuing name')
	
	return parser


def verify(ver_model_1, ver_model_2, ques, new_ans, tik_encoding, overseer_model = None):
	res1, error1, og_pred1, num_pred1, ip_toks, op_toks = check_model_output(ques, new_ans, ver_model_1, "8-shot-cot", 2048, 0.5, [], tik_encoding)

	tot_ip_toks = ip_toks
	tot_op_toks = op_toks

	if error1:
		return 0, ques, new_ans, tot_ip_toks, tot_op_toks

	if not res1:
		return 0, ques, new_ans, tot_ip_toks, tot_op_toks

	res2, error2, og_pred1, num_pred1, ip_toks, op_toks = check_model_output(ques, new_ans, ver_model_2, "8-shot-cot", 2048, 0.5, [], tik_encoding)

	tot_ip_toks += ip_toks
	tot_op_toks += op_toks

	if error2:
		return 0, ques, new_ans, tot_ip_toks, tot_op_toks
	
	if not res2:
		return 0, ques, new_ans, tot_ip_toks, tot_op_toks
	
	if overseer_model is not None:
		res3, error3, og_pred3, num_pred3, ip_toks, op_toks = check_model_output(ques, new_ans, overseer_model, "8-shot-cot", 2048, 0.5, [], tik_encoding)

		tot_ip_toks += ip_toks
		tot_op_toks += op_toks

		if error3:
			return 0, ques, new_ans, tot_ip_toks, tot_op_toks
		if not res3:
			return 0, ques, new_ans, tot_ip_toks, tot_op_toks
		
		return res3, ques, new_ans, tot_ip_toks, tot_op_toks

	return res2, ques, new_ans, tot_ip_toks, tot_op_toks


def grammar_correct(model, context1, context2, context3):
	prompt, sys_prompt = get_verification_prompt("grammar_correct", params=(context1, context2, context3))

	og_pred = model.predict(prompt, sys_prompt, 1024, 0.5, 1, [])

	error = False

	try:
		corr_context1, corr_context2, corr_context3 = process_grammar_corrected(og_pred)
	except:
		error = True
		corr_context1 = context1
		corr_context2 = context2
		corr_context3 = context3

	return error, corr_context1, corr_context2, corr_context3


def problem_extend_generation(args, seed_data, model, ver_model_1, ver_model_2, overseer_model, grammar_model, prompt_type, num_iters=3, max_tokens=8000, temperature=1.0, stop=[], tik_encoding=None):
	problems = []
	done_seed_ques = []
	if args.continue_name != "default":
		problems_data = pd.read_csv(args.out_dir_name + "/" + args.continue_name + "/problems.tsv", sep = '\t')
		# convert to list of dicts
		for prob in problems_data.to_dict(orient='records'):
			problems.append([prob["question"], prob["answer"], prob["depth"], prob["seed question"], prob["seed answer"], json.loads(prob["contexts"]), json.loads(prob["reasonings"]), json.loads(prob["intermediate answers"])])
			done_seed_ques.append(prob["seed question"])
		print("Loaded {} problems...".format(len(problems)))

	num_done = len(problems)

	st_flag = True

	example_start_time = -1
	for i in range(len(seed_data)):
		seed_ques = seed_data.loc[i]["question"]
		seed_ans = seed_data.loc[i]["answer"]
		seed_ans_num = float(seed_ans.replace(",", "").split("####")[-1].strip())

		if seed_ques in done_seed_ques:
			num_done -= 1
			continue
		elif num_done > 0:
			continue
		elif st_flag:
			print("Starting from: ", i+1)
			st_flag = False

		if example_start_time > 0:
			print("Time taken for previous example: ", time.time() - example_start_time)
		
		example_start_time = time.time()

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Original Question: " + str(seed_ques) + "\n\n")
			f.write("Original Answer: " + str(seed_ans) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------\n\n")
		
		contexts = []
		reasonings = []
		final_ques_stmt = ""
		final_answer = 0

		successful_iters = 0

		ver_init_ans_stmt = ""
		ver_init_context = ""
		ver_mid_context = ""
		ver_ques_stmt = ""

		ls_ans_stmt = []

		ver_trip_init_ans_stmt = ""

		ans_to_avoid = [seed_ans_num]

		iter_start_time = -1
		creation_ip_toks = 0
		creation_op_toks = 0

		verification_ip_toks = 0
		verification_op_toks = 0

		for j in range(num_iters):
			if iter_start_time > 0:
				print("Time taken for iteration {}: {}".format(j, time.time() - iter_start_time))
			iter_start_time = time.time()

			prompt, sys_prompt = get_generator_prompt(prompt_type, params=(seed_ques, seed_ans))

			cur_time = time.time()
			og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)
			# print("Time taken for model prediction: ", time.time() - cur_time)

			creation_ip_toks += len(tik_encoding.encode(prompt))
			creation_op_toks += len(tik_encoding.encode(og_pred, disallowed_special=()))

			try:
				prev_ans_stmt, og_context, new_context, new_ques_stmt, new_ans, new_reasoning = process_output(prompt_type, og_pred)
			except:
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Failure to Parse...\n\n")
					f.write("Prediction:\n" + str(og_pred) + "\n\n")
				continue

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Seed Question: " + str(seed_ques) + "\n\n")
				f.write("Seed Answer: " + str(seed_ans) + "\n\n")
				f.write("------------------------------------------------\n\n")
				f.write("Full Prediction:\n" + str(og_pred) + "\n\n")
				f.write("------------------------------------------------\n\n")
				f.write("Original context: " + str(og_context) + "\n\n")
				f.write("Original Answer Statement: " + str(prev_ans_stmt) + "\n\n")
				f.write("------------------------------------------------\n\n")
				f.write("New context: " + str(new_context) + "\n\n")
				f.write("New question statement: " + str(new_ques_stmt) + "\n\n")
				f.write("New answer reasoning: " + str(new_reasoning) + "\n\n")
				f.write("New answer: " + str(new_ans) + "\n\n")

			prev_ans_flag = False
			ans_to_check = ans_to_avoid[-1]
			if str(ans_to_check) not in new_reasoning:
				if int(ans_to_check) == ans_to_check:
					if str(int(ans_to_check)) not in new_reasoning:
						prev_ans_flag = True
				else:
					prev_ans_flag = True
			
			if prev_ans_flag:
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("----------------Previous Answer Check Step----------------\n\n")
					f.write("Previous answer not found in new reasoning...\n\n")
					f.write("Previous Answer: " + str(ans_to_check) + "\n\n")
					f.write("New Reasoning: " + str(new_reasoning) + "\n\n")
					f.write("--------------------------------------------------------------------------------------------\n\n")
				continue

			og_context = nums_check(og_context, ans_to_avoid)
			new_context = nums_check(new_context, ans_to_avoid)

			if " how " in og_context.lower() and len(og_context.lower().split(" how ")) > 1:
				if "?" in og_context.lower().split(" how ")[-1]:
					og_context = "how".join(og_context.split(" how ")[:-1]).strip()
					og_context = "How".join(og_context.split(" How ")[:-1]).strip()

			if " how " in new_context.lower() and len(new_context.lower().split(" how ")) > 1:
				if "?" in new_context.lower().split(" how ")[-1]:
					if "how" not in new_ques_stmt.lower() and "?" not in new_ques_stmt.lower(): 
						new_ques_stmt = " how " + new_context.lower().split(" how ")[-1] + new_ques_stmt
					new_context = "how".join(new_context.split(" how ")[:-1]).strip()
					new_context = "How".join(new_context.split(" How ")[:-1]).strip()

			ques_to_verify = prev_ans_stmt + " " + new_context + " " + new_ques_stmt

			cur_time = time.time()
			if args.indi_verify:
				ver_res, ver_ques, ver_ans, ver_ip_toks, ver_op_toks = verify(ver_model_1, ver_model_2, ques_to_verify, new_ans, tik_encoding)

				verification_ip_toks += ver_ip_toks
				verification_op_toks += ver_op_toks
			# print("Time taken for verification 1: ", time.time() - cur_time)

			if len(contexts) == 0:
				ver_init_context = og_context
				ver_mid_context = new_context
				ver_ques_stmt = new_ques_stmt
				double_ver_ques_to_verify = ver_init_ans_stmt + " " + ver_init_context + " " + ver_mid_context + " " + ver_ques_stmt
				double_ver_ques_to_verify = double_ver_ques_to_verify.strip()

				cur_time = time.time()
				double_ver_res, double_ver_ques, double_ver_ans, ver_ip_toks, ver_op_toks = verify(ver_model_1, ver_model_2, double_ver_ques_to_verify, new_ans, tik_encoding)
				# print("Time taken for verification 2: ", time.time() - cur_time)

				verification_ip_toks += ver_ip_toks
				verification_op_toks += ver_op_toks

			if len(contexts) > 0:
				cur_time = time.time()
				if args.grammar:
					grammar_error, corr_context1, corr_context2, corr_context3 = grammar_correct(grammar_model, contexts[-2], contexts[-1], new_context)
				else:
					grammar_error = -1
					corr_context1 = contexts[-2]
					corr_context2 = contexts[-1]
					corr_context3 = new_context
				# print("Time taken for grammar correction: ", time.time() - cur_time)

				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("----------------Grammar Step----------------\n\n")
					f.write("Grammar Correction Error: " + str(grammar_error) + "\n\n")
					f.write("Corrected Context 1: " + str(corr_context1) + "\n\n")
					f.write("Corrected Context 2: " + str(corr_context2) + "\n\n")
					f.write("Corrected Context 3: " + str(corr_context3) + "\n\n")
					f.write("--------------------------------------------------------------------------------------------\n\n")

				ver_ques_stmt = new_ques_stmt.strip()
				if len(ls_ans_stmt) > 1:
					ver_trip_init_ans_stmt = ls_ans_stmt[-2]

				ver_main_context = corr_context1 + " " + corr_context2 + " " + corr_context3

				triple_ver_ques_to_verify = ver_trip_init_ans_stmt + " " + ver_main_context + " " + ver_ques_stmt
				triple_ver_ques_to_verify = triple_ver_ques_to_verify.strip()

				cur_time = time.time()
				if args.overseer:
					triple_ver_res, triple_ver_ques, triple_ver_ans, ver_ip_toks, ver_op_toks = verify(ver_model_1, ver_model_2, triple_ver_ques_to_verify, new_ans, tik_encoding, overseer_model)
				else:
					triple_ver_res, triple_ver_ques, triple_ver_ans, ver_ip_toks, ver_op_toks = verify(ver_model_1, ver_model_2, triple_ver_ques_to_verify, new_ans, tik_encoding)
				# print("Time taken for verification 2: ", time.time() - cur_time)

				verification_ip_toks += ver_ip_toks
				verification_op_toks += ver_op_toks

			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("----------------Verification Step----------------\n\n")
				if args.indi_verify:
					f.write("Verification Question: " + str(ver_ques) + "\n\n")
					f.write("Verification Answer: " + str(new_ans) + "\n\n")
					f.write("Verification Result: " + str(ver_res) + "\n\n")
				if len(contexts) == 0:
					f.write("Double Verification Question: " + str(double_ver_ques) + "\n\n")
					f.write("Verification Answer: " + str(new_ans) + "\n\n")
					f.write("Double Verification Result: " + str(double_ver_res) + "\n\n")
				if len(contexts) > 0:
					f.write("Triple Verification Question: " + str(triple_ver_ques) + "\n\n")
					f.write("Verification Answer: " + str(new_ans) + "\n\n")
					f.write("Triple Verification Result: " + str(triple_ver_res) + "\n\n")
				f.write("--------------------------------------------------------------------------------------------\n\n")

			if args.indi_verify and not ver_res:
				# print("Verification Failed...")
				# print("Question was: ", ver_ques)
				# print("Answer was: ", ver_ans)
				# print()
				continue
			if not double_ver_res:
				# print("Double Verification Failed...")
				# print("Question was: ", double_ver_ques)
				# print("Answer was: ", double_ver_ans)
				# print()
				continue
			if len(contexts) > 0 and not triple_ver_res:
				# print("Triple Verification Failed...")
				# print("Question was: ", triple_ver_ques)
				# print("Answer was: ", triple_ver_ans)
				# print()
				continue

			if len(contexts) > 0:
				new_context = corr_context3
				contexts[-1] = corr_context2
				contexts[-2] = corr_context1

			if len(contexts) == 0:
				contexts = [og_context]
				reasonings = [seed_ans.split("####")[0].strip()]

			contexts.append(new_context)
			reasonings.append(new_reasoning)

			repeated_phrase = find_longest_repeated_phrase(contexts[-2] + " " + contexts[-1], 8)
			if len(repeated_phrase) > 0:
				# print("Repeated phrase found...")
				# print("Phrase: ", repeated_phrase)
				# print()
				with open(args.out_dir + "/logs.txt", "a") as f:
					f.write("Repeated phrase found...\n\n")
					f.write("contexts[-2]: " + str(contexts[-2]) + "\n\n")
					f.write("contexts[-1]: " + str(contexts[-1]) + "\n\n")
					f.write("Phrase: " + str(repeated_phrase) + "\n\n")
				
				contexts[-1] = contexts[-1].replace(repeated_phrase, "").strip()

			ans_to_avoid.append(new_ans)

			seed_ques = prev_ans_stmt + " " + new_context + " " + new_ques_stmt
			seed_ans = new_reasoning + "\n#### " + str(new_ans)

			ls_ans_stmt.append(prev_ans_stmt)

			final_ques_stmt = new_ques_stmt
			final_answer = new_ans

			successful_iters += 1
			if successful_iters >= args.max_depth:
				break
		
		if successful_iters < args.min_depth:
			continue

		final_question = " ".join(contexts) + " " + final_ques_stmt
		final_answer = " ".join(reasonings) + "\n####" + str(final_answer)
			
		problems.append([final_question, final_answer, successful_iters, seed_data.loc[i]["question"], seed_data.loc[i]["answer"], contexts, reasonings, ans_to_avoid])

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Final Question: " + str(final_question) + "\n\n")
			f.write("Final Answer: " + str(final_answer) + "\n\n")
			f.write("Successful Iters: " + str(successful_iters) + "\n\n")
			f.write("Creation IP Toks: " + str(creation_ip_toks) + "\n\n")
			f.write("Creation OP Toks: " + str(creation_op_toks) + "\n\n")
			f.write("Verification IP Toks: " + str(verification_ip_toks) + "\n\n")
			f.write("Verification OP Toks: " + str(verification_op_toks) + "\n\n")
			f.write("========================================================================================================\n\n")
			f.write("========================================================================================================\n\n")

		print("{} successful problems created so far... out of {}".format(len(problems), i+1))

		pred_df = pd.DataFrame(problems, columns = ['question', 'answer', 'depth', 'seed question', 'seed answer', 'contexts', 'reasonings', 'intermediate answers'])
		pred_df['contexts'] = pred_df['contexts'].apply(json.dumps)
		pred_df['reasonings'] = pred_df['reasonings'].apply(json.dumps)
		pred_df['intermediate answers'] = pred_df['intermediate answers'].apply(json.dumps)
		pred_df.to_csv(args.out_dir + "/problems.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(seed_data)), end = '\r', flush = True)

	print("Successfully generated {} problems...".format(len(problems)))


def naive_generation(args, seed_data, model, prompt_type, max_tokens=8000, temperature=1.0, stop=[], tik_encoding=None):
	problems = []

	for i in range(len(seed_data)):
		seed_ques = seed_data.loc[i]["question"]
		seed_ans = seed_data.loc[i]["answer"]
		seed_ans_num = float(seed_ans.replace(",", "").split("####")[-1].strip())

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("Original Question: " + str(seed_ques) + "\n\n")
			f.write("Original Answer: " + str(seed_ans) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------\n\n")

		prompt, sys_prompt = get_generator_prompt(prompt_type, params=(seed_ques, seed_ans))

		og_pred = model.predict(prompt, sys_prompt, max_tokens, temperature, 1, stop)

		try:
			new_ques, new_ans = process_naive(og_pred)
		except:
			print("Failure to Parse...")
			with open(args.out_dir + "/logs.txt", "a") as f:
				f.write("Failure to Parse...\n\n")
				f.write("Prediction:\n" + str(og_pred) + "\n\n")
			continue

		with open(args.out_dir + "/logs.txt", "a") as f:
			f.write("New question: " + str(new_ques) + "\n\n")
			f.write("New answer: " + str(new_ans) + "\n\n")
		
		problems.append([new_ques, new_ans])

		print("{} successful problems created so far... out of {}".format(len(problems), i+1))

		pred_df = pd.DataFrame(problems, columns = ['question', 'answer'])
		pred_df.to_csv(args.out_dir + "/naive_problems.tsv", sep = '\t', index = None)

		print("Completed {} / {}...".format(i+1, len(seed_data)), end = '\r', flush = True)

	print("Successfully generated {} problems...".format(len(problems)))


def main(args):
	try:
		tik_encoding = tiktoken.encoding_for_model(args.model)
	except:
		tik_encoding = tiktoken.encoding_for_model("gpt-4")

	_, sys_prompt = get_generator_prompt(args.prompt_type, params=("dummy", "dummy", "dummy", "dummy", "dummy", "dummy"))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	ver_model_1 = LargeLanguageModel(model_type=args.ver_model_type_1, model=args.ver_model_1, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	ver_model_2 = LargeLanguageModel(model_type=args.ver_model_type_2, model=args.ver_model_2, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	overseer_model = LargeLanguageModel(model_type=args.overseer_model_type, model=args.overseer_model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	grammar_model = LargeLanguageModel(model_type=args.grammar_model_type, model=args.grammar_model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	if args.exp_type == "problem_extend":
		seed_data = pd.read_csv("data/" + args.seed_name + ".tsv", sep = '\t')
		problem_extend_generation(args, seed_data, model, ver_model_1, ver_model_2, overseer_model, grammar_model, args.prompt_type, args.num_iters, args.max_tokens, args.temperature, args.stop, tik_encoding)
	elif args.exp_type == "naive_baseline":
		seed_data = pd.read_csv("data/" + args.seed_name + ".tsv", sep = '\t')
		naive_generation(args, seed_data, model, args.prompt_type, args.max_tokens, args.temperature, args.stop, tik_encoding)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.exp_type + "_" + args.model + "_" + str(args.temperature) +  "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)