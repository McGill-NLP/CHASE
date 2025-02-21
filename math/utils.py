import pdb
import re
from word2number import w2n
from collections import Counter

# Regular expression pattern to match number phrases in words
# This pattern matches numbers and relevant conjunctions like 'and' (e.g., "twenty-five", "two hundred and ten")
number_pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|and|[-\s])+\b'

def nums_check(text, nums):
	new_text = text
	# Find all potential number phrases in the text
	matches = re.findall(number_pattern, text)

	# Convert each number phrase to numeric form using word2number
	dict_matches = {}
	converted_numbers = []
	for match in matches:
		try:
			number = w2n.word_to_num(match)
			converted_numbers.append(number)
			dict_matches[number] = match
		except ValueError:
			# Skip any non-number words or invalid matches
			pass
	
	for num in nums:
		if num in converted_numbers:
			new_text = new_text.replace(dict_matches[num], "")
	
		new_text = new_text.replace(str(num), "")
		if num == int(num):
			new_text = new_text.replace(str(int(num)), "")

	return new_text


def find_longest_repeated_phrase(text, min_phrase_length=8):
	# Split the text into words
	words = text.split()
	
	max_phrase_length = len(words)  # The longest possible phrase is the entire text

	# Start with the minimum phrase length and incrementally check longer phrases
	longest_repeated_phrase = ""
	for phrase_length in range(min_phrase_length, max_phrase_length + 1):
		# Create phrases of `phrase_length` words using a sliding window
		phrases = [' '.join(words[i:i+phrase_length]) for i in range(len(words) - phrase_length + 1)]

		# Count occurrences of each phrase
		phrase_counts = Counter(phrases)

		# Find any repeated phrases of this length
		repeated_phrases = [phrase for phrase, count in phrase_counts.items() if count > 1]

		# If there are repeated phrases, update the longest repeated phrase
		if repeated_phrases:
			longest_repeated_phrase = repeated_phrases[0]  # Take the first repeated phrase (all should be the same length)
		else:
			# If no repeated phrases are found at this length, stop searching
			break

	return longest_repeated_phrase


def process_grammar_corrected(output):
	lines = output.split("\n")

	for line_no in range(len(lines)):
		if lines[line_no].strip().startswith("Corrected context 1:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Corrected context 2:"):
					end_line = temp_no
					break
			corr_context1 = "\n".join(lines[line_no:end_line]).strip().split("Corrected context 1:")[1].strip()
			if corr_context1[0] == "<" and corr_context1[-1] == ">":
				corr_context1 = corr_context1[1:-1]
		elif lines[line_no].strip().startswith("Corrected context 2:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("Corrected context 3:"):
					end_line = temp_no
					break
			corr_context2 = "\n".join(lines[line_no:end_line]).strip().split("Corrected context 2:")[1].strip()
			if corr_context2[0] == "<" and corr_context2[-1] == ">":
				corr_context2 = corr_context2[1:-1]
		elif lines[line_no].strip().startswith("Corrected context 3:"):
			corr_context3 = "\n".join(lines[line_no:len(lines)]).strip().split("Corrected context 3:")[1].strip()
			if corr_context3[0] == "<" and corr_context3[-1] == ">":
				corr_context3 = corr_context3[1:-1]
			break
	
	return corr_context1, corr_context2, corr_context3


def process_output(prompt_type, output):
	output = output.replace("**", "")
	output = output.replace("##", "")

	if prompt_type == "problem_extend":
		lines = output.split("\n")

		for line_no in range(len(lines)):
			if lines[line_no].strip().startswith("Original context [without question statement]:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Question statement:"):
						end_line = temp_no
						break
				og_context = "\n".join(lines[line_no:end_line]).strip().split("Original context [without question statement]:")[1].strip()
				if og_context[0] == "<" and og_context[-1] == ">":
					og_context = og_context[1:-1]
			elif lines[line_no].strip().startswith("Original context:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Question statement:"):
						end_line = temp_no
						break
				og_context = "\n".join(lines[line_no:end_line]).strip().split("Original context:")[1].strip()
				if og_context[0] == "<" and og_context[-1] == ">":
					og_context = og_context[1:-1]
			elif lines[line_no].strip().startswith("Original answer statement:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New operation over original answer"):
						end_line = temp_no
						break
				prev_ans_stmt = "\n".join(lines[line_no:end_line]).strip().split("Original answer statement:")[1].strip()
				if prev_ans_stmt[0] == "<" and prev_ans_stmt[-1] == ">":
					prev_ans_stmt = prev_ans_stmt[1:-1]
			elif lines[line_no].strip().startswith("New context [Do not mention original answer]:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New question statement:"):
						end_line = temp_no
						break
				new_context = "\n".join(lines[line_no:end_line]).strip().split("New context [Do not mention original answer]:")[1].strip()
				if new_context[0] == "<" and new_context[-1] == ">":
					new_context = new_context[1:-1]
			elif lines[line_no].strip().startswith("New context:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New question statement:"):
						end_line = temp_no
						break
				new_context = "\n".join(lines[line_no:end_line]).strip().split("New context:")[1].strip()
				if new_context[0] == "<" and new_context[-1] == ">":
					new_context = new_context[1:-1]
			elif lines[line_no].strip().startswith("New question statement:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New answer reasoning:"):
						end_line = temp_no
						break
				new_ques_stmt = "\n".join(lines[line_no:end_line]).strip().split("New question statement:")[1].strip()
				if new_ques_stmt[0] == "<" and new_ques_stmt[-1] == ">":
					new_ques_stmt = new_ques_stmt[1:-1]
			elif lines[line_no].strip().startswith("New answer reasoning:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New answer [Number only]:") or lines[temp_no].strip().startswith("New answer:"):
						end_line = temp_no
						break
				reasoning = "\n".join(lines[line_no:end_line]).strip().split("New answer reasoning:")[1].strip()
				if reasoning[0] == "<" and reasoning[-1] == ">":
					reasoning = reasoning[1:-1]
			elif lines[line_no].strip().startswith("New answer [Number only]:"):
				new_ans = "\n".join(lines[line_no:len(lines)]).strip().split("New answer [Number only]:")[1].strip().split()[0].replace("$", "").replace(",", "").replace("%", "")
				if new_ans[0] == "<" and new_ans[-1] == ">":
					new_ans = new_ans[1:-1]
				new_ans = float(new_ans)
			elif lines[line_no].strip().startswith("New answer:"):
				new_ans = "\n".join(lines[line_no:len(lines)]).strip().split("New answer:")[1].strip().split()[0].replace("$", "").replace(",", "").replace("%", "")
				if new_ans[0] == "<" and new_ans[-1] == ">":
					new_ans = new_ans[1:-1]
				new_ans = float(new_ans)

		new_context = new_context.replace(new_ques_stmt, "")

		return prev_ans_stmt, og_context, new_context, new_ques_stmt, new_ans, reasoning
	
	elif prompt_type == "problem_extend_small_model":
		output = output.split("seed question:")[0].strip()
		output = output.split("Seed question:")[0].strip()
		lines = output.split("\n")

		for line_no in range(len(lines)):
			if lines[line_no].strip().startswith("Original context"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Question statement:"):
						end_line = temp_no
						break
				og_context = "\n".join(lines[line_no:end_line]).strip().split(":")[1].strip()
				if og_context[0] == "<" and og_context[-1] == ">":
					og_context = og_context[1:-1]
			elif lines[line_no].strip().startswith("Original answer statement"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New operation over original answer"):
						end_line = temp_no
						break
				prev_ans_stmt = "\n".join(lines[line_no:end_line]).strip().split(":")[1].strip()
				if prev_ans_stmt[0] == "<" and prev_ans_stmt[-1] == ">":
					prev_ans_stmt = prev_ans_stmt[1:-1]
			elif lines[line_no].strip().startswith("New context"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New question statement"):
						end_line = temp_no
						break
				new_context = "\n".join(lines[line_no:end_line]).strip().split(":")[1].strip()
				if new_context[0] == "<" and new_context[-1] == ">":
					new_context = new_context[1:-1]
			elif lines[line_no].strip().startswith("New question statement"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New answer reasoning"):
						end_line = temp_no
						break
				new_ques_stmt = "\n".join(lines[line_no:end_line]).strip().split(":")[1].strip()
				if new_ques_stmt[0] == "<" and new_ques_stmt[-1] == ">":
					new_ques_stmt = new_ques_stmt[1:-1]
			elif lines[line_no].strip().startswith("New answer reasoning"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New answer") or lines[temp_no].strip().startswith("New answer:"):
						end_line = temp_no
						break
				reasoning = "\n".join(lines[line_no:end_line]).strip().split(":")[1].strip()
				if reasoning[0] == "<" and reasoning[-1] == ">":
					reasoning = reasoning[1:-1]
			elif lines[line_no].strip().startswith("New answer"):
				new_ans = "\n".join(lines[line_no:len(lines)]).strip().split(":")[1].strip().split()[0].replace("$", "").replace(",", "").replace("%", "")
				if new_ans[0] == "<" and new_ans[-1] == ">":
					new_ans = new_ans[1:-1]
				new_ans = float(new_ans)

		new_context = new_context.replace(new_ques_stmt, "")

		return prev_ans_stmt, og_context, new_context, new_ques_stmt, new_ans, reasoning
	
	elif prompt_type == "problem_bottom_up_break":
		lines = output.split("\n")

		qtys = []

		for line_no in range(len(lines)):
			if lines[line_no].strip().startswith("Original context [without question statement]:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Question statement:"):
						end_line = temp_no
						break
				og_context = "\n".join(lines[line_no:end_line]).strip().split("Original context [without question statement]:")[1].strip()
				if og_context[0] == "<" and og_context[-1] == ">":
					og_context = og_context[1:-1]
			if lines[line_no].strip().startswith("Question statement:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Answer statement:"):
						end_line = temp_no
						break
				ques_stmt = "\n".join(lines[line_no:end_line]).strip().split("Question statement:")[1].strip()
				if ques_stmt[0] == "<" and ques_stmt[-1] == ">":
					ques_stmt = ques_stmt[1:-1]
			if lines[line_no].strip().startswith("Answer statement:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Quantity "):
						end_line = temp_no
						break
				ans_stmt = "\n".join(lines[line_no:end_line]).strip().split("Answer statement:")[1].strip()
				if ques_stmt[0] == "<" and ques_stmt[-1] == ">":
					ques_stmt = ques_stmt[1:-1]

			if lines[line_no].strip().startswith("Number chosen [present in original question, is to be asked by new question]:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Modified original context [without including the chosen number & without question statement]:"):
						end_line = temp_no
						break
				prev_ans = "\n".join(lines[line_no:end_line]).strip().split("Number chosen [present in original question, is to be asked by new question]:")[1].strip().split()[0].replace("$", "").replace(",", "").replace("%", "")
				if prev_ans[0] == "<" and prev_ans[-1] == ">":
					prev_ans = prev_ans[1:-1]
				if "/" in prev_ans:
					numerator = float(prev_ans.split("/")[0])
					denominator = float(prev_ans.split("/")[1])
					prev_ans = numerator / denominator
				prev_ans = float(prev_ans)

				for tlno in range(line_no+1, len(lines)):
					if lines[tlno].strip().startswith("Quantity "):
						break
					if lines[tlno].strip().startswith("Modified original context [without including the chosen number & without question statement]:"):
						end_line = tlno + 1
						for temp_no in range(tlno + 1, len(lines)):
							if lines[temp_no].strip().startswith("New question statement based on chosen quantity [comprehensive, mentioning subjects and objects]:"):
								end_line = temp_no
								break
						modified_context = "\n".join(lines[tlno:end_line]).strip().split("Modified original context [without including the chosen number & without question statement]:")[1].strip()
						if modified_context[0] == "<" and modified_context[-1] == ">":
							modified_context = modified_context[1:-1]
					elif lines[tlno].strip().startswith("New question statement based on chosen quantity [comprehensive, mentioning subjects and objects]:"):
						end_line = tlno + 1
						for temp_no in range(tlno + 1, len(lines)):
							if lines[temp_no].strip().startswith("Subjects involved in new question statement:"):
								end_line = temp_no
								break
						new_ques_stmt = "\n".join(lines[tlno:end_line]).strip().split("New question statement based on chosen quantity [comprehensive, mentioning subjects and objects]:")[1].strip()
						if new_ques_stmt[0] == "<" and new_ques_stmt[-1] == ">":
							new_ques_stmt = new_ques_stmt[1:-1]
					elif lines[tlno].strip().startswith("Subjects involved in new question statement:"):
						end_line = tlno + 1
						for temp_no in range(tlno + 1, len(lines)):
							if lines[temp_no].strip().startswith("Objects involved in new question statement:"):
								end_line = temp_no
								break
						subjs = "\n".join(lines[tlno:end_line]).strip().split("Subjects involved in new question statement:")[1].strip()
						if subjs[0] == "<" and subjs[-1] == ">":
							subjs = subjs[1:-1]
					elif lines[tlno].strip().startswith("Objects involved in new question statement:"):
						end_line = tlno + 1
						for temp_no in range(tlno + 1, len(lines)):
							if lines[temp_no].strip().startswith("Quantity "):
								end_line = temp_no
								break
						objs = "\n".join(lines[tlno:end_line]).strip().split("Objects involved in new question statement:")[1].strip()
						if objs[0] == "<" and objs[-1] == ">":
							objs = objs[1:-1]
				qtys.append((modified_context, new_ques_stmt, prev_ans, subjs, objs))

		return og_context, ques_stmt, ans_stmt, qtys
	
	elif prompt_type == "problem_bottom_up_create":
		lines = output.split("\n")

		for line_no in range(len(lines)):
			if lines[line_no].strip().startswith("Complete new problem (context + question statement):"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("Context information in new problem [without any question statement]:"):
						end_line = temp_no
						break
				new_ques = "\n".join(lines[line_no:end_line]).strip().split("Complete new problem (context + question statement):")[1].strip()
				if new_ques[0] == "<" and new_ques[-1] == ">":
					new_ques = new_ques[1:-1]
			elif lines[line_no].strip().startswith("Context information in new problem [without any question statement]:"):
				end_line = line_no + 1
				for temp_no in range(line_no + 1, len(lines)):
					if lines[temp_no].strip().startswith("New reasoning:"):
						end_line = temp_no
						break
				new_context = "\n".join(lines[line_no:end_line]).strip().split("Context information in new problem [without any question statement]:")[1].strip()
				if new_context[0] == "<" and new_context[-1] == ">":
					new_context = new_context[1:-1]
			elif lines[line_no].strip().startswith("New reasoning:"):
				new_reasoning = "\n".join(lines[line_no:len(lines)]).strip().split("New reasoning:")[1].strip()
				if new_reasoning[0] == "<" and new_reasoning[-1] == ">":
					new_reasoning = new_reasoning[1:-1]

		return new_ques, new_context, new_reasoning


def process_naive(output):
	output = output.replace("**", "")
	output = output.replace("##", "")
	
	lines = output.split("\n")

	for line_no in range(len(lines)):
		if lines[line_no].strip().startswith("Rewritten problem:"):
			end_line = line_no + 1
			for temp_no in range(line_no + 1, len(lines)):
				if lines[temp_no].strip().startswith("New answer:"):
					end_line = temp_no
					break
			ques = "\n".join(lines[line_no:end_line]).strip().split("Rewritten problem:")[1].strip()
			if ques[0] == "<" and ques[-1] == ">":
				ques = ques[1:-1]
		elif lines[line_no].strip().startswith("New answer:"):
			new_ans = "\n".join(lines[line_no:len(lines)]).strip().split("New answer:")[1].strip()
			if new_ans[0] == "<" and new_ans[-1] == ">":
				new_ans = new_ans[1:-1]

	return ques, new_ans