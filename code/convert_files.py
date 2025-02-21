import os
import json
import pandas as pd
import tiktoken
import argparse

def build_parser():
	parser = argparse.ArgumentParser(description='Convert files to readable format')

	parser.add_argument('-data_dir', type=str, default='data/', help='Output Directory')
	parser.add_argument('-data', type=str, default='chase_code_dp', help='Data filename')
	parser.add_argument('-path', type=str, default='readable_data/', help='Output Directory')
	parser.add_argument('-example', type=str, default='all', help='Which example to convert')
	
	return parser

def main(args):
	data = pd.read_csv(args.data_dir + args.data + ".tsv", sep="\t")

	tik_enc = tiktoken.encoding_for_model("gpt-4o")

	if not os.path.exists(args.path):
		os.makedirs(args.path)

	max_context_size = 0
	avg_context_size = 0

	cnt = 0

	for i in range(len(data)):
		if args.example != "all":
			if i != int(args.example):
				continue

		context = data.loc[i]["Context"]
		problem = data.loc[i]["Problem"]
		ans = json.loads(data.loc[i]["Answer"])
		test = data.loc[i]["Test"]

		main_cont = context + problem
		context_size = len(tik_enc.encode(main_cont))
		
		avg_context_size += context_size
		cnt += 1

		if context_size > max_context_size:
			max_context_size = context_size

		cur_path = args.path + "/" + str(i) + ".txt"

		with open(cur_path, "a") as f:
			f.write("Question No: " + str(i) + "\n")
			f.write("Context Size: " + str(context_size) + "\n\n")
			f.write("Codebase:\n\n" + str(context) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------------\n\n")
			f.write("Problem Statement: " + str(problem) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------------\n\n")
			f.write("Answer Code:\n" + str(ans["function_def"]) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------------\n\n")
			f.write("Test Code:\n" + str(test) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------------\n\n")

	if avg_context_size == 0:
		print("Please enter a valid example number between 0 and " + str(len(data) - 1))
	else:
		print("Max Context Size: ", str(max_context_size))
		print("Avg Context Size: ", str(avg_context_size / cnt))


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	main(args)