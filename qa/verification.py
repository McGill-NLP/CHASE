import os
import re
import argparse
import pandas as pd
import pdb
import json
import openai
import anthropic
import google.generativeai as genai

from models import LargeLanguageModel
from prompts import get_verification_prompt

def build_parser():
	parser = argparse.ArgumentParser(description='Verify')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, default='programmatic_docs_gpt-4o-mini_1.0_2024-09-16-16:54:42_0', help='Folder name')
	parser.add_argument('-data', type=str, default='programmatic_data_modified', help='Data filename')
	parser.add_argument('-exp_type', type=str, default='programmatic_docs', help='Exp type')


	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-model_type', type=str, default='vllm', choices=['completion', 'chat', 'vllm', 'gemini', 'peft', 'anthropic'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=8000, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.5, help='Sampling temperature')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') # Alter this or temp, not both
	parser.add_argument('-n', type=int, default=1, help='number of completions to generate for each prompt')
	parser.add_argument('-presence_penalty', type=float, default=0.0, help='positive values increases model\'s likelihood to talk about new topics')
	parser.add_argument('-frequency_penalty', type=float, default=0.0, help='positive values decreases model\'s likelihood to repeat same line verbatim')
	
	return parser


def get_doc_list(docs):
	pattern = r"(Document \d+:)"

	# Split the documents by the pattern, keeping the pattern as a delimiter
	split_docs = re.split(pattern, docs)

	# Remove the first element if it's empty
	if len(split_docs)%2 == 1:
		split_docs = split_docs[1:]

	###########################################CHANGEEEEEEEEEEEE##############################################
	# Reconstruct the documents by combining the header with the following content
	doc_list = [split_docs[i] + "\n" + split_docs[i+1].split("Answer points assigned:")[0].strip() + "\nText:\n"  + split_docs[i+1].split("Text:")[-1].strip() for i in range(0, len(split_docs), 2)]

	return doc_list

def presence_check(args, model, ques, doc_ans_pts, rel_doc_list):
	check_flag = True
	to_add = {}
	for doc_no in range(len(rel_doc_list)):
		ans_pts = doc_ans_pts[str(doc_no + 1)]
		doc = rel_doc_list[doc_no]

		add_pts = []

		for pt in ans_pts:
			prompt, sys_prompt = get_verification_prompt("presence", params=(ques, pt, doc))
			og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
			presence_res = False
			if "true" in og_pred.split("Explanation for Presence")[0].lower():
				presence_res = True
			relevance_res = False
			if "true" in og_pred.split("Explanation for Relevance")[0].split("Relevance:")[1].lower():
				relevance_res = True
			if presence_res == False or relevance_res == False:
				check_flag = False
				add_pts.append(pt)

			with open(args.verification_dir + "/presence_logs.txt", "a") as f:
				f.write(str(doc) + "\n\n")
				f.write("Answer Point: " + str(pt) + "\n\n")
				f.write("Prediction:\n" + og_pred + "\n")
				f.write("---------------------------------------------------------\n")

		to_add[str(doc_no + 1)] = add_pts
	return check_flag, to_add


def extra_check_relevant(args, model, ques, doc_ans_pts, doc_list):
	check_flag = True
	to_remove = {}

	avoid_str = [
		"none of the points in the document",
		"document does not provide any information"
	]

	for doc_no in range(len(doc_list)):
		ans_pts = doc_ans_pts[str(doc_no + 1)]
		doc = doc_list[doc_no]

		answer_pts_str = "\n".join(ans_pts)

		prompt, sys_prompt = get_verification_prompt("extra", params=(ques, answer_pts_str, doc))
		og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
		res = False
		if "false" in og_pred.split("Extra Points Mentioned")[0].lower():
			res = True
		if res == False:
			extra_pts_mentioned = og_pred.split("Extra Points Mentioned")[1].split(":")[1].strip()
			for av_str in avoid_str:
				if av_str in extra_pts_mentioned.lower():
					res = True
					break
			if res == False:
				to_remove[str(doc_no + 1)] = extra_pts_mentioned
				check_flag = False

		with open(args.verification_dir + "/extra_logs.txt", "a") as f:
			f.write(str(doc) + "\n\n")
			f.write("Answer: " + str(answer_pts_str) + "\n\n")
			f.write("Prediction:\n" + og_pred + "\n")
			f.write("---------------------------------------------------------\n")

	return check_flag, to_remove


def extra_check_adversarial(args, model, ques, meta_doc_list):
	check_flag = True
	meta_to_remove = []
	for doc_list in meta_doc_list:
		to_remove = {}
		for doc_no in range(len(doc_list)):
			doc = doc_list[doc_no]

			prompt, sys_prompt = get_verification_prompt("extra_adv", params=(ques, doc))
			og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
			res = False
			if "false" in og_pred.split("Relevant Points")[0].lower():
				res = True
			if res == False:
				to_remove[str(doc_no + 1)] = og_pred.split("Relevant Points")[1].split(":")[1].strip()
				check_flag = False

			with open(args.verification_dir + "/extra_logs_adv.txt", "a") as f:
				f.write(str(doc) + "\n\n")
				f.write("Prediction:\n" + og_pred + "\n")
				f.write("---------------------------------------------------------\n")
		meta_to_remove.append(to_remove)

	return check_flag, meta_to_remove


def pred_check_relevant(args, model, ques, doc_ans_pts, doc_list):
	check_flag = True
	to_remove = {}

	for doc_no in range(len(doc_list)):
		ans_pts = doc_ans_pts[str(doc_no + 1)]
		doc = doc_list[doc_no]

		answer_pts_str = "\n".join(ans_pts)

		prompt, sys_prompt = get_verification_prompt("predict", params=(ques, doc))
		model_answer = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)

		fname = "/pred_logs.txt"

		with open(args.verification_dir + fname, "a") as f:
			f.write(str(doc) + "\n\n")
			f.write("Prediction: " + str(model_answer) + "\n\n")

		if model_answer.lower().strip() != "no relevant information found in this document.":
			prompt, sys_prompt = get_verification_prompt("compare", params=(ques, answer_pts_str, model_answer))
			og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)

			res = False
			if "false" in og_pred.split("Extra Points Mentioned")[0].lower():
				res = True
			if res == False:
				extra_pts_mentioned = og_pred.split("Extra Points Mentioned")[1].split(":")[1].strip()
				to_remove[str(doc_no + 1)] = extra_pts_mentioned
				check_flag = False

			with open(args.verification_dir + fname, "a") as f:
				f.write("Ground Truth Answer Points: " + str(answer_pts_str) + "\n\n")
				f.write("Comparison Result:\n" + og_pred + "\n")
				f.write("---------------------------------------------------------\n")

	return check_flag, to_remove


def pred_check_adversarial(args, model, ques, meta_doc_list):
	check_flag = True
	meta_to_remove = []

	for doc_list in meta_doc_list:
		to_remove = {}
		for doc_no in range(len(doc_list)):
			doc = doc_list[doc_no]

			prompt, sys_prompt = get_verification_prompt("predict", params=(ques, doc))
			model_answer = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)

			fname = "/pred_logs_adv.txt"

			with open(args.verification_dir + fname, "a") as f:
				f.write(str(doc) + "\n\n")
				f.write("Prediction: " + str(model_answer) + "\n\n")
				f.write("---------------------------------------------------------\n")

			if model_answer.lower().strip() != "no relevant information found in this document.":
				to_remove[str(doc_no + 1)] = model_answer
				check_flag = False

		meta_to_remove.append(to_remove)

	return check_flag, meta_to_remove


def remove_extra(args, model, ques, doc_ans_pts, to_remove, doc_list):
	new_doc_list = []
	for doc_no in range(len(doc_list)):
		doc = doc_list[doc_no]
		if str(doc_no + 1) not in to_remove:
			new_doc_list.append(doc)
			continue
		if doc_ans_pts is not None:
			ans_pts = doc_ans_pts[str(doc_no + 1)]
			answer_pts_str = "\n".join(ans_pts)
		else:
			answer_pts_str = None
		info_to_remove = to_remove[str(doc_no + 1)]

		prompt, sys_prompt = get_verification_prompt("remove", params=(ques, answer_pts_str, info_to_remove, doc))
		og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
		
		new_doc_list.append(og_pred.strip())

		with open(args.verification_dir + "/remove_logs.txt", "a") as f:
			f.write(str(doc) + "\n\n")
			f.write("Answer: " + str(answer_pts_str) + "\n\n")
			f.write("Info to remove:\n" + info_to_remove + "\n")
			f.write("Corrected Document:\n" + str(og_pred) + "\n\n")
			f.write("---------------------------------------------------------\n")
	return new_doc_list


def add_info(args, model, ques, to_add, doc_list):
	new_doc_list = []
	for doc_no in range(len(doc_list)):
		doc = doc_list[doc_no]
		if len(to_add[str(doc_no + 1)]) == 0:
			new_doc_list.append(doc)
			continue
		
		info_to_add = "\n".join(to_add[str(doc_no + 1)])

		prompt, sys_prompt = get_verification_prompt("add", params=(ques, info_to_add, doc))
		og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
		
		new_doc_list.append(og_pred.strip())

		with open(args.verification_dir + "/add_logs.txt", "a") as f:
			f.write(str(doc) + "\n\n")
			f.write("Info to add:\n" + info_to_add + "\n")
			f.write("Corrected Document:\n" + str(og_pred) + "\n\n")
			f.write("---------------------------------------------------------\n")
	return new_doc_list


def programmatic_docs_verify(args, data):
	ls = data.to_dict(orient='records')
	new_ls = []

	if os.path.exists(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv"):
		temp_new_df = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv", sep="\t")
		new_ls = temp_new_df.to_dict(orient='records')
	
	start_idx = len(new_ls)

	_, sys_prompt = get_verification_prompt("presence", params=("", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	args.verification_dir = args.out_dir + "/" + args.folder_name + "/" + "verification_logs"
	if not os.path.exists(args.verification_dir):
		os.makedirs(args.verification_dir)

	print("Starting at: ", str(start_idx))

	for i in range(start_idx, len(ls)):
		ques = ls[i]["Question"]
		ans = ls[i]["Answer"]

		rel_doc_list = json.loads(ls[i]["Rel_Docs_List"])
		adv_doc_list = json.loads(ls[i]["Adv_Docs_List"])

		doc_ans_pts = json.loads(ls[i]["Doc_Ans_Points"])
		ans_pts = json.loads(ls[i]["Ans_Points"])

		num_keys = len(doc_ans_pts.keys())
		if num_keys != len(rel_doc_list):
			rel_doc_list = rel_doc_list[:num_keys]
			with open(args.verification_dir + "/extra_logs.txt", "a") as f:
				f.write("Number of Docs and Number of ans points mismatched!\n")
				f.write("Question: " + str(ques) + "\n")
				f.write("------------------------------------------------------------------------------\n")
				if num_keys > len(rel_doc_list):
					for key_no in range(len(rel_doc_list)+1, num_keys+1):
						ans_pts_to_del = doc_ans_pts.pop(str(key_no), None)
						for apt in ans_pts_to_del:
							ans_pts.remove(apt)
							ans = ans.replace(apt, "").strip()
							ans = ans.replace("\n- \n-", "\n-").strip()

		########################################################################################################
		# Check if there is any point in the docs that should have been in the answer. Remove them from docs.

		with open(args.verification_dir + "/extra_logs.txt", "a") as f:
			f.write("Question " + str(i+1) + ": " + ques + "\n")

		extra, to_remove = extra_check_relevant(args, model, ques, doc_ans_pts, rel_doc_list)

		with open(args.verification_dir + "/extra_logs.txt", "a") as f:
			f.write("Final Extra Check: " + str(extra) + "\n\n")
			f.write("======================================================================================\n")

		temp_rel_doc_list = remove_extra(args, model, ques, doc_ans_pts, to_remove, rel_doc_list)

		########################################################################################################
		# Get model to predict answer and see if there is anything extra relevant. Remove them from docs.

		with open(args.verification_dir + "/pred_logs.txt", "a") as f:
			f.write("Question " + str(i+1) + ": " + ques + "\n")

		pred_extra, pred_to_remove = pred_check_relevant(args, model, ques, doc_ans_pts, temp_rel_doc_list)

		with open(args.verification_dir + "/pred_logs.txt", "a") as f:
			f.write("Final Pred Check: " + str(pred_extra) + "\n\n")
			f.write("======================================================================================\n")

		new_rel_doc_list = remove_extra(args, model, ques, doc_ans_pts, pred_to_remove, temp_rel_doc_list)

		########################################################################################################
		# Check if there is any point in the adv docs that is relevant for the question. Remove them from adv docs.

		with open(args.verification_dir + "/extra_logs_adv.txt", "a") as f:
			f.write("Question " + str(i+1) + ": " + ques + "\n")

		adv_extra, meta_adv_to_remove = extra_check_adversarial(args, model, ques, adv_doc_list)

		with open(args.verification_dir + "/extra_logs_adv.txt", "a") as f:
			f.write("Final Extra Check: " + str(adv_extra) + "\n\n")
			f.write("======================================================================================\n")

		temp_adv_doc_list = []
		for az in range(len(meta_adv_to_remove)):
			cur_adv_doc_list = remove_extra(args, model, ques, None, meta_adv_to_remove[az], adv_doc_list[az])
			temp_adv_doc_list.append(cur_adv_doc_list)

		########################################################################################################
		# Get model to predict answer and see if there is anything extra relevant in adv docs. Remove them from adv docs.

		with open(args.verification_dir + "/pred_logs_adv.txt", "a") as f:
			f.write("Question " + str(i+1) + ": " + ques + "\n")

		adv_pred_extra, meta_adv_pred_to_remove = pred_check_adversarial(args, model, ques, temp_adv_doc_list)

		with open(args.verification_dir + "/pred_logs_adv.txt", "a") as f:
			f.write("Final Pred Check: " + str(adv_pred_extra) + "\n\n")
			f.write("======================================================================================\n")

		new_adv_doc_list = []
		for az in range(len(meta_adv_pred_to_remove)):
			cur_adv_doc_list = remove_extra(args, model, ques, None, meta_adv_pred_to_remove[az], temp_adv_doc_list[az])
			new_adv_doc_list.append(cur_adv_doc_list)

		########################################################################################################
		# Check presence, and relevance of assigned answer points in documents.

		with open(args.verification_dir + "/presence_logs.txt", "a") as f:
			f.write("Question " + str(i+1) + ": " + ques + "\n")

		presence, to_add = presence_check(args, model, ques, doc_ans_pts, new_rel_doc_list)

		with open(args.verification_dir + "/presence_logs.txt", "a") as f:
			f.write("Final Presence Check: " + str(presence) + "\n\n")
			f.write("======================================================================================\n")

		final_rel_doc_list = add_info(args, model, ques, to_add, new_rel_doc_list)

		########################################################################################################

		ls[i]["Rel_Docs_List"] = final_rel_doc_list
		ls[i]["Adv_Docs_List"] = new_adv_doc_list
		ls[i]["Ans_Points"] = json.dumps(ans_pts)
		ls[i]["Doc_Ans_Points"] = json.dumps(doc_ans_pts)
		ls[i]["Answer"] = ans

		new_ls.append(ls[i])

		print("Completed {} / {}...".format(i+1, len(ls)), end = '\r', flush = True)

		new_df = pd.DataFrame(new_ls)

		new_df['Rel_Docs_List'] = new_df['Rel_Docs_List'].apply(json.dumps)
		new_df['Adv_Docs_List'] = new_df['Adv_Docs_List'].apply(json.dumps)

		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv", sep = '\t', index = None)
	
	print("\nVerified Data: ", len(new_df))


def adv_cross_check(args, model, ques, pt):
	check_flag = True
	
	prompt, sys_prompt = get_verification_prompt("adv_cross_check", params=(ques, pt))
	og_pred = model.predict(prompt, sys_prompt, args.max_tokens, args.temperature, 1, args.stop)
	
	if "true" in og_pred.split("Relevance:")[1].lower():
		check_flag = False

	with open(args.verification_dir + "/logs.txt", "a") as f:
		f.write("Question: " + str(ques) + "\n")
		f.write("Answer Point: " + str(pt) + "\n")
		f.write("Prediction: " + str(og_pred) + "\n")
		f.write("======================================================================================\n")
		
	return check_flag


def create_docs_info(doc_ans_pts):
	doc_info = ""
	for doc_no in doc_ans_pts:
		doc_info = doc_info + "Document " + str(doc_no) + " Answer points assigned:\n"
		for pt in doc_ans_pts[doc_no]:
			doc_info = doc_info + "- " + pt + "\n"
		doc_info = doc_info + "\n"
	return doc_info.strip()


def programmatic_adversarial_verify(args, data):
	ls = data.to_dict(orient='records')
	new_ls = []

	if os.path.exists(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv"):
		temp_new_df = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv", sep="\t")
		new_ls = temp_new_df.to_dict(orient='records')
	
	start_idx = len(new_ls)

	_, sys_prompt = get_verification_prompt("presence", params=("", "", ""))

	model = LargeLanguageModel(model_type=args.model_type, model=args.model, peft_model="none", sys_prompt=sys_prompt, top_p=args.top_p, presence_penalty=args.presence_penalty, frequency_penalty=args.frequency_penalty)

	args.verification_dir = args.out_dir + "/" + args.folder_name + "/" + "verification_logs"
	if not os.path.exists(args.verification_dir):
		os.makedirs(args.verification_dir)

	print("Starting at: ", str(start_idx))

	num_pairs = 0
	num_examples = 0

	for i in range(start_idx, len(ls)):
		ques = json.loads(ls[i]["Questions"])
		ans = json.loads(ls[i]["Answers"])
		docs_info = json.loads(ls[i]["Documents_Info"])
		ans_pts = json.loads(ls[i]["Ans_Points"])
		doc_ans_pts = json.loads(ls[i]["Doc_Ans_Points"])

		idx_to_remove = []
		num_data = len(ques)

		for j in range(len(ans_pts)):
			og_answer = ans[j]
			ques_to_check = [ques[z] for z in range(len(ques)) if z != j]
			ans_pts_to_check = ans_pts[j].copy()
			for k in range(len(ans_pts_to_check)):
				cur_ans_pt = ans_pts_to_check[k]
				valid = True
				for l in range(len(ques_to_check)):
					cur_ques = ques_to_check[l]
					res = adv_cross_check(args, model, cur_ques, cur_ans_pt)
					if res == False:
						valid = False
						break
				if valid == False:
					ans_pts[j].remove(cur_ans_pt)
					doc_ans_pts[j] = dict(sorted(doc_ans_pts[j].items(), key=lambda x: int(x[0])))
					dict_keys = list(doc_ans_pts[j].keys())
					break_flag = False
					for doc_no in dict_keys:
						if cur_ans_pt in doc_ans_pts[j][doc_no]:
							doc_ans_pts[j][doc_no].remove(cur_ans_pt)
							break_flag = True
						if len(doc_ans_pts[j][doc_no]) == 0:
							del doc_ans_pts[j][doc_no]
						if break_flag:
							break
					doc_ans_pts[j] = dict(sorted(doc_ans_pts[j].items(), key=lambda x: int(x[0])))
					dict_keys = list(doc_ans_pts[j].keys())
					for doc_key in range(len(dict_keys)):
						doc_ans_pts[j][str(doc_key+1)] = doc_ans_pts[j].pop(dict_keys[doc_key])
					ans[j] = ans[j].replace(cur_ans_pt, "").strip()
					ans[j] = ans[j].replace("- \n", "").strip()

			if len(ans_pts[j]) == 0:
				idx_to_remove.append(j)
				num_pairs += 1
				with open(args.verification_dir + "/logs.txt", "a") as f:
					f.write("\n\n----------------------------------REMOVAL OF PAIR----------------------------\n")
					for t_q in ques_to_check:
						f.write("Question: " + str(t_q) + "\n")
					f.write("\nAnswer:\n" + str(og_answer) + "\n")
					f.write("======================================================================================\n\n\n")
			else:
				doc_ans_pts[j] = dict(sorted(doc_ans_pts[j].items(), key=lambda x: int(x[0])))
				docs_info[j] = create_docs_info(doc_ans_pts[j])
				with open(args.verification_dir + "/logs.txt", "a") as f:
					f.write("\n----------------------------------NEW DOCS INFO----------------------------\n")
					f.write("Docs Info:\n" + str(docs_info[j]) + "\n")
					f.write("======================================================================================\n\n")
		
		if len(idx_to_remove) != num_data:
			with open(args.verification_dir + "/logs.txt", "a") as f:
				f.write("\n\nIndices to remove: " + str(idx_to_remove) + "\n-----------------------------------------\n\n")

			for idx in sorted(idx_to_remove, reverse=True):
				del ques[idx]
				del ans[idx]
				del docs_info[idx]
				del ans_pts[idx]
				del doc_ans_pts[idx]

			ls[i]["Questions"] = json.dumps(ques)
			ls[i]["Answers"] = json.dumps(ans)
			ls[i]["Documents_Info"] = json.dumps(docs_info)
			ls[i]["Ans_Points"] = json.dumps(ans_pts)
			ls[i]["Doc_Ans_Points"] = json.dumps(doc_ans_pts)

			new_ls.append(ls[i])
			new_df = pd.DataFrame(new_ls)
			new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_verified.tsv", sep = '\t', index = None)
		else:
			num_examples += 1
			with open(args.verification_dir + "/logs.txt", "a") as f:
				f.write("\n\n----------------------------------REMOVAL OF DATA POINT!----------------------------\n")
				for t_q in ques:
					f.write("Question: " + str(t_q) + "\n")
				f.write("======================================================================================\n\n\n")

		print("Completed {} / {}...".format(i+1, len(ls)), end = '\r', flush = True)

	print("Removed {} pairs and {} examples...".format(num_pairs, num_examples))
	
	print("\nVerified Data: ", len(new_df))


def main(args):
	if args.exp_type == "programmatic_docs":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_docs_verify(args, data)
	elif args.exp_type == "programmatic_adversarial":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_adversarial_verify(args, data)
	
if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	openai.api_key = os.getenv("OPENAI_API_KEY")
	genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
	anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

	main(args)