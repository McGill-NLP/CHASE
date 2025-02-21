import os
import pandas as pd
import tiktoken
import argparse

def build_parser():
	parser = argparse.ArgumentParser(description='Convert files to readable format')

	parser.add_argument('-data_dir', type=str, default='data/', help='Output Directory')
	parser.add_argument('-data', type=str, default='chase_qa', help='Data filename')
	parser.add_argument('-path', type=str, default='readable_data/', help='Output Directory')
	parser.add_argument('-example', type=str, default='all', help='Which example to convert')
	
	return parser

def main(args):
	data = pd.read_csv(args.data_dir + "/" + args.data + ".tsv", sep='\t')

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
		
		cnt += 1
		
		qno = data.loc[i]["Question_No"]
		ques = data.loc[i]["Question"]
		ans = data.loc[i]["Answer"]
		docs = data.loc[i]["Documents"]

		context = docs + ques
		context_size = len(tik_enc.encode(context))

		avg_context_size += context_size
		
		if context_size > max_context_size:
			max_context_size = context_size

		cur_path = args.path + str(qno) + ".txt"

		with open(cur_path, "a") as f:
			f.write("Question No: " + str(qno) + "\n")
			f.write("Context Size: " + str(context_size) + "\n\n")
			f.write("Documents:\n\n" + str(docs) + "\n\n")
			f.write("--------------------------------------------------------------------------------------------------\n\n")
			f.write("Question: " + str(ques) + "\n\n")
			f.write("Answer:\n" + str(ans) + "\n\n")
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