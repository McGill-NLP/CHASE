import argparse
import pandas as pd
import pdb
import json

def build_parser():
	parser = argparse.ArgumentParser(description='Post Process')

	parser.add_argument('-out_dir', type=str, default='generation_outputs/', help='Output Directory')
	parser.add_argument('-folder_name', type=str, default='gpt-4o-mini-qa', help='Folder name')
	parser.add_argument('-data', type=str, default='prog_qa', help='Data filename')
	parser.add_argument('-exp_type', type=str, default='programmatic_qa', help='Exp type')
	parser.add_argument('-verbose', type=bool, default=False, help='Verbose')
	
	return parser

def programmatic_qa_process(data):
	ls = data.to_dict(orient='records')
	exceptions_ls = []
	new_ls = []

	for i in range(len(ls)):
		try:
			answer = ls[i]["Answer"]

			ans_points_og = answer.split("\n")
			ans_points = []
			for ans_pt in ans_points_og:
				if ans_pt[0] == "-":
					ans_points.append(ans_pt[1:].strip())
				else:
					ans_points.append(ans_pt.strip())

			ans_points_copy = ans_points.copy()

			docs_info = ls[i]["Documents_Info"]

			doc_ans_points = {1: []}
			doc_no = 1
			for line in docs_info.split("\n"):
				if len(line) > 2:
					if line[:8] == "Document":
						if line[9] != str(doc_no):
							doc_no += 1
							doc_ans_points[doc_no] = []
						if "title:" not in line.lower():
							if len(line.strip().split(":")[1]) > 2:
								pt_candidate = line.strip().split(":")[1].strip()
								if pt_candidate[0] == "-":
									doc_ans_points[doc_no].append(pt_candidate[1:].strip())
									ans_points_copy.remove(pt_candidate[1:].strip())
								else:
									doc_ans_points[doc_no].append(pt_candidate)
									ans_points_copy.remove(pt_candidate)
					else:
						if line.strip()[0] == "-":
							doc_ans_points[doc_no].append(line.strip()[1:].strip())
							ans_points_copy.remove(line.strip()[1:].strip())
						else:
							doc_ans_points[doc_no].append(line.strip())
							ans_points_copy.remove(line.strip())

			if len(ans_points_copy) > 0:
				raise Exception("Some points did not match!")

			ls[i]["Ans_Points"] = ans_points
			ls[i]["Doc_Ans_Points"] = doc_ans_points

			new_ls.append(ls[i])
		except Exception as e:
			if args.verbose:
				print("At index: ", str(i))
				print("Question: ", ls[i]["Question"])
				print("Answer:\n", ls[i]["Answer"])
				print("Documents Info:\n", ls[i]["Documents_Info"])
				print("Exception: ", str(e))
				print("Note: list remove errors occur when there is a mismatch between the answer points and the document answer points. The expectation is that all points in the answer should be assigned (written again) in the document answer points.")
				print()
			exceptions_ls.append(ls[i])
			continue

	if len(new_ls) > 0:
		new_df = pd.DataFrame(new_ls)
		try:
			new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
			new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
		except:
			pass
		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None)

	if len(exceptions_ls) > 0:
		exc_df = pd.DataFrame(exceptions_ls)
		exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep = '\t', index = None)

def programmatic_adversarial_process(data):
	ls = data.to_dict(orient='records')
	exceptions_ls = []
	new_ls = []

	for i in range(len(ls)):
		try:
			ls_ans_pts = []
			ls_doc_ans_pts = []
			og_answer = json.loads(ls[i]["Adv_Answer"])

			for j in range(len(og_answer)):
				answer = og_answer[j]

				ans_points_og = answer.split("\n")
				ans_points = []
				for ans_pt in ans_points_og:
					if ans_pt[0] == "-":
						ans_points.append(ans_pt[1:].strip())
					else:
						ans_points.append(ans_pt.strip())

				ans_points_copy = ans_points.copy()

				docs_info = json.loads(ls[i]["Adv_Documents_Info"])[j]

				doc_ans_points = {1: []}
				doc_no = 1
				for line in docs_info.split("\n"):
					if len(line) > 2:
						if line[:8] == "Document":
							if line[9] != str(doc_no):
								doc_no += 1
								doc_ans_points[doc_no] = []
							if "title" not in line.lower():
								if len(line.strip().split(":")[1]) > 2:
									pt_candidate = line.strip().split(":")[1].strip()
									if pt_candidate[0] == "-":
										doc_ans_points[doc_no].append(pt_candidate[1:].strip())
										ans_points_copy.remove(pt_candidate[1:].strip())
									else:
										doc_ans_points[doc_no].append(pt_candidate)
										ans_points_copy.remove(pt_candidate)
						else:
							if line.strip()[0] == "-":
								doc_ans_points[doc_no].append(line.strip()[1:].strip())
								ans_points_copy.remove(line.strip()[1:].strip())
							else:
								doc_ans_points[doc_no].append(line.strip())
								ans_points_copy.remove(line.strip())

				if len(ans_points_copy) > 0:
					raise Exception("Some points did not match!")

				ls_ans_pts.append(ans_points)
				ls_doc_ans_pts.append(doc_ans_points)

			ls[i]["Adv_Ans_Points"] = ls_ans_pts
			ls[i]["Adv_Doc_Ans_Points"] = ls_doc_ans_pts

			new_ls.append(ls[i])
		except Exception as e:
			if args.verbose:
				print("At index: ", str(i))
				print("Adv Question: ", ls[i]["Adv_Question"])
				print("Adv Answer:\n", ls[i]["Adv_Answer"])
				print("Adv Documents Info:\n", ls[i]["Adv_Documents_Info"])
				print("Exception: ", str(e))
				print()
			exceptions_ls.append(ls[i])
			continue

	if len(new_ls) > 0:
		final_ls = []

		for i in range(len(new_ls)):
			id1 = new_ls[i]["ID"]
			persona = new_ls[i]["Persona"]
			env = new_ls[i]["Environment"]
			sim = new_ls[i]["Similarity"]
			adv_questions = json.loads(new_ls[i]["Adv_Question"])
			adv_answers = json.loads(new_ls[i]["Adv_Answer"])
			adv_docs_info = json.loads(new_ls[i]["Adv_Documents_Info"])
			adv_ans_pts = new_ls[i]["Adv_Ans_Points"]
			adv_doc_ans_pts = new_ls[i]["Adv_Doc_Ans_Points"]

			questions_ls = [new_ls[i]["Question"]]
			answers_ls = [new_ls[i]["Answer"]]
			docs_info_ls = [new_ls[i]["Documents_Info"]]
			ans_pts_ls = [json.loads(new_ls[i]["Ans_Points"])]
			doc_ans_pts_ls = [json.loads(new_ls[i]["Doc_Ans_Points"])]


			for j in range(len(adv_questions)):
				questions_ls.append(adv_questions[j])
				answers_ls.append(adv_answers[j])
				docs_info_ls.append(adv_docs_info[j])
				ans_pts_ls.append(adv_ans_pts[j])
				doc_ans_pts_ls.append(adv_doc_ans_pts[j])

			final_ls.append([id1, persona, env, sim, questions_ls, answers_ls, docs_info_ls, ans_pts_ls, doc_ans_pts_ls])

		new_df = pd.DataFrame(final_ls, columns = ['ID', 'Persona', 'Environment', 'Similarity', 'Questions', 'Answers', 'Documents_Info', 'Ans_Points', 'Doc_Ans_Points'])
		new_df['Questions'] = new_df['Questions'].apply(json.dumps)
		new_df['Answers'] = new_df['Answers'].apply(json.dumps)
		new_df['Documents_Info'] = new_df['Documents_Info'].apply(json.dumps)
		new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
		new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
		new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None)
		print("Length of Final Data: ", str(len(new_df)))

	if len(exceptions_ls) > 0:
		exc_df = pd.DataFrame(exceptions_ls)
		exc_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_exceptions.tsv", sep = '\t', index = None)


def programmatic_docs_process(data):
	ls = data.to_dict(orient='records')
	new_ls = []
	
	tot_cnt = 1

	for i in range(len(ls)):
		id1 = ls[i]["ID"]
		persona = ls[i]["Persona"]
		env = ls[i]["Environment"]
		questions = json.loads(ls[i]["Questions"])
		answers = json.loads(ls[i]["Answers"])
		docs_info = json.loads(ls[i]["Documents_Info"])
		ans_pts = json.loads(ls[i]["Ans_Points"])
		doc_ans_pts = json.loads(ls[i]["Doc_Ans_Points"])
		docs_list = json.loads(ls[i]["Docs_List"])

		modified_docs_list = []
		for doc_ls in docs_list:
			new_doc_ls = []
			for doc in doc_ls:
				if "Title:" in doc:
					mod_doc = "Title: " + doc.split("Title:")[1].strip()
				mod_doc = mod_doc.split("In conclusion,")[0].strip()
				mod_doc = mod_doc.split("In summary,")[0].strip()
				mod_doc = mod_doc.split("To summarize")[0].strip()
				new_doc_ls.append(mod_doc)
			modified_docs_list.append(new_doc_ls)

		for j in range(len(questions)):
			cur_ques = questions[j]
			cur_ans = answers[j]
			cur_ans_pts = ans_pts[j]
			cur_doc_ans_pts = doc_ans_pts[j]
			cur_docs = modified_docs_list[j]

			adv_ques = questions.copy()
			adv_ques.remove(cur_ques)
			adv_ans = answers.copy()
			adv_ans.remove(cur_ans)
			adv_ans_pts = ans_pts.copy()
			adv_ans_pts.remove(cur_ans_pts)
			adv_doc_ans_pts = doc_ans_pts.copy()
			adv_doc_ans_pts.remove(cur_doc_ans_pts)
			adv_docs_list = modified_docs_list.copy()
			adv_docs_list.remove(cur_docs)

			new_ls.append([id1, tot_cnt, persona, env, cur_ques, cur_ans, cur_ans_pts, cur_doc_ans_pts, cur_docs, adv_ques, adv_ans, adv_ans_pts, adv_doc_ans_pts, adv_docs_list])
			
			tot_cnt += 1
	
	new_df = pd.DataFrame(new_ls, columns = ['Root_ID', 'Question_No', 'Persona', 'Environment', 'Question', 'Answer', 'Ans_Points', 'Doc_Ans_Points', 'Rel_Docs_List', 'Adv_Question', 'Adv_Answer', 'Adv_Ans_Pts', 'Adv_Doc_Ans_Pts', 'Adv_Docs_List'])
	new_df['Ans_Points'] = new_df['Ans_Points'].apply(json.dumps)
	new_df['Doc_Ans_Points'] = new_df['Doc_Ans_Points'].apply(json.dumps)
	new_df['Rel_Docs_List'] = new_df['Rel_Docs_List'].apply(json.dumps)
	new_df['Adv_Question'] = new_df['Adv_Question'].apply(json.dumps)
	new_df['Adv_Answer'] = new_df['Adv_Answer'].apply(json.dumps)
	new_df['Adv_Ans_Pts'] = new_df['Adv_Ans_Pts'].apply(json.dumps)
	new_df['Adv_Doc_Ans_Pts'] = new_df['Adv_Doc_Ans_Pts'].apply(json.dumps)
	new_df['Adv_Docs_List'] = new_df['Adv_Docs_List'].apply(json.dumps)
	new_df.to_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + "_modified.tsv", sep = '\t', index = None)
	print("Length of Final Data: ", str(len(new_df)))

	
def main(args):
	if args.exp_type == "programmatic_qa":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_qa_process(data)
	elif args.exp_type == "programmatic_adversarial":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_adversarial_process(data)
	elif args.exp_type == "programmatic_docs":
		data = pd.read_csv(args.out_dir + "/" + args.folder_name + "/" + args.data + ".tsv", sep='\t')
		programmatic_docs_process(data)
	
if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	main(args)