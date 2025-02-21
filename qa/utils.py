import random

def sample_scenarios(ls, num_samples=5):
	text = ""
	sampled_ls = random.sample(ls, num_samples)
	for i in range(len(sampled_ls)):
		persona = sampled_ls[i][1]
		env = sampled_ls[i][2]
		text += "USER_PERSONA: " + persona + "\n"
		text += "COLLECTION_OF_DOCS: " + env + "\n\n"
	return text.strip()

def process_naive(output):
	output = output.replace("**", "")
	output = output.replace("##", "")

	lines = output.split("\n")

	for line_no in range(len(lines)):
		if lines[line_no].strip().startswith("Documents:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Question:"):
					end_line = temp_no
					break
			docs = "\n".join(lines[line_no:end_line]).strip().split("Documents:")[1].strip()
			if docs[0] == "<" and docs[-1] == ">":
				docs = docs[1:-1]
		elif lines[line_no].strip().startswith("Question:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Answer:"):
					end_line = temp_no
					break
			ques = "\n".join(lines[line_no:end_line]).strip().split("Question:")[1].strip()
			if ques[0] == "<" and ques[-1] == ">":
				ques = ques[1:-1]
		elif lines[line_no].strip().startswith("Answer:"):
			ans = "\n".join(lines[line_no:len(lines)]).strip().split("Answer:")[1].strip()
			if ans[0] == "<" and ans[-1] == ">":
				ans = ans[1:-1]

	return docs, ques, ans