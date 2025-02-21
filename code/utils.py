import json
import random


standard_libs = ["sklearn", "typing", "pandas", "scipy", "itertools", "datetime", "collections", "statistics"]
standard_funcs = ["len", "range", "enumerate", "zip", "sum", "min", "max", "abs", "round", "sorted"]

def jaccard_similarity(set1, set2):
	# Calculate the intersection of the two sets
	intersection = set1.intersection(set2)
	
	# Calculate the union of the two sets
	union = set1.union(set2)
	
	# Calculate the Jaccard similarity coefficient
	if not union:
		return 100.0  # If both sets are empty, we define the similarity as 100%
	similarity = (len(intersection) / len(union)) * 100
	
	return similarity


def get_functions(text):
	text = text.replace("```Python", "").replace("```", "")
	lines = text.split("\n")
	funcs = []
	for line_no in range(len(lines)):
		if "Parameters:" in lines[line_no]:
			params = {}
			end_line = line_no + 1
			for temp_no in range(line_no+1, len(lines)):
				if "Objectives:" in lines[temp_no]:
					end_line = temp_no
					break
				elif len(lines[temp_no]) > 1 and lines[temp_no].strip()[0] == "-":
					para_line = lines[temp_no].strip()[1:].strip().split(":")
					if len(para_line) > 1:
						params[para_line[0].strip()] = para_line[1].strip()
			objectives = []
			for temp_no in range(end_line+1, len(lines)):
				if "Function" in lines[temp_no] and "in file" in lines[temp_no]:
					end_line = temp_no
					break
				elif len(lines[temp_no]) > 1 and lines[temp_no].strip()[0] == "-":
					obj_line = lines[temp_no].strip()[1:].strip()
					objectives.append(obj_line)
			function_name = lines[end_line].split("in file")[0].split("Function")[1].replace('"','').replace("'","").strip()
			file_name = lines[end_line].split("in file")[1].split(":")[0].replace('"','').replace("'","").strip()
			import_lines = []
			for temp_no in range(end_line+1, len(lines)):
				if "def " in lines[temp_no]:
					end_line = temp_no
					break
				elif "import " in lines[temp_no]:
					import_lines.append(lines[temp_no].strip())
			function_def = ""
			start_line = end_line
			subfunc_flag = False
			for temp_no in range(end_line+1, len(lines)):
				if lines[temp_no].startswith("\tdef ") or lines[temp_no].startswith("    def "):
					subfunc_flag = True
					break
				if lines[temp_no].startswith("\treturn ") or lines[temp_no].startswith("    return "):
					end_line = temp_no
					break
				if lines[temp_no][0] != " " and lines[temp_no][0] != "\t":
					end_line = temp_no-1
					break
			if subfunc_flag:
				continue
			function_def = "\n".join(lines[start_line:end_line+1])
			if len(lines) >= end_line + 1:
				thresh = 4
				if len(lines) < end_line + thresh:
					thresh = len(lines) - end_line
				if "def " in lines[end_line + 1:end_line + thresh]:
					continue
			funcs.append(
				{
					"function_name": function_name,
					"file_name": file_name,
					"parameters": params,
					"objectives": objectives,
					"import_lines": import_lines,
					"function_def": function_def
				}
			)
	return funcs


def get_adversarial_functions(text, file_name):
	text = remove_python_wrapper(text)
	lines = text.split("\n")
	import_lines = []
	defs = []
	for line_no in range(len(lines)):
		if "import " in lines[line_no]:
			import_lines.append(lines[line_no].strip())
		elif "def " in lines[line_no]:
			start_line = line_no
			end_line = -1
			for temp_no in range(start_line+1, len(lines)):
				if "def " in lines[temp_no]:
					end_line = temp_no
					break
			if end_line == -1:
				end_line = len(lines)
			defs.append("\n".join(lines[start_line:end_line]).strip())

	funcs = []

	for fn_def in defs:
		funcs.append(
			{
				"function_name": fn_def.split("(")[0].split("def ")[1].strip(),
				"file_name": file_name,
				"import_lines": import_lines,
				"function_def": fn_def
			}
		)

	return funcs


def obtain_functions_text(functions):
	functions_txt = ""
	for function in functions:
		functions_txt += "Parameters:\n"
		for param in function["parameters"]:
			param_line = "- " + param + ": " + function["parameters"][param] + "\n"
			functions_txt += param_line
		functions_txt += "Objectives:\n"
		for obj in function["objectives"]:
			obj_line = "- " + obj + "\n"
			functions_txt += obj_line
		functions_txt += "\nFunction " + '"' + function["function_name"] + '"' + " in file " + '"' + function["file_name"] + '"' + ":\n\n"
		for imp_line in function["import_lines"]:
			functions_txt += imp_line + "\n"
		functions_txt += "\n" + function["function_def"] + "\n\n"
	return functions_txt


def obtain_adv_functions_text(functions):
	functions_txt = ""
	for function in functions:
		functions_txt += "Function " + '"' + function["function_name"] + '"' + " in file " + '"' + function["file_name"] + '"' + ":\n\n"
		for imp_line in function["import_lines"]:
			functions_txt += imp_line + "\n"
		functions_txt += "\n" + function["function_def"] + "\n\n"
	return functions_txt


def obtain_main_fn_txt(ans_code):
	main_fn_txt = ""
	main_fn_txt += "Function " + ans_code["function_name"] + " in file " + ans_code["file_name"] + ":\n\n"
	for imp_line in ans_code["import_lines"]:
		main_fn_txt += imp_line + "\n"
	main_fn_txt += "\n" + ans_code["function_def"]

	return main_fn_txt


def obtain_seed_functions(domain, num_funcs=3):
	seed_functions_txt = ""
	with open("helper_functions/generated/" + domain + ".json", "r") as f:
		seed_functions = json.load(f)
		seed_functions = random.sample(seed_functions, num_funcs)
		seed_functions_txt = obtain_functions_text(seed_functions)
	return seed_functions, seed_functions_txt


def get_objectives(problem):
	lines = problem.split("\n")
	objectives = []
	for line_no in range(len(lines)):
		if "Objectives:" in lines[line_no]:
			for temp_no in range(line_no+1, len(lines)):
				if "Return Values:" in lines[temp_no]:
					break
				elif len(lines[temp_no]) > 1 and lines[temp_no].strip()[0] == "-":
					obj_line = lines[temp_no].strip()
					objectives.append(obj_line)

	return "\n".join(objectives).strip()


def correct_import_line(import_line):
	return import_line.replace("from .", "from ").replace("from C.", "from ").replace("from Codebase.", "from ").replace("from codebase.", "from ").strip()


def remove_python_wrapper(text):
	if "```Python" in text:
		pred_code = text.split("```Python")[1].split("```")[0].strip()
	else:
		pred_code = text
	if "```python" in pred_code:
		pred_code = pred_code.split("```python")[1].split("```")[0].strip()
	
	return pred_code.replace(".py import", " import")


def process_output(text):
	if text.strip().startswith("```Python"):
		text = text.split("```Python")[1].strip().replace(".py:\n", ".py:\n\n```Python\n")
	if text.strip().startswith("```python"):
		text = text.split("```python")[1].strip().replace(".py:\n", ".py:\n\n```python\n")
	lines = text.split("\n")
	for line_no in range(len(lines)):
		if 'Function ' == lines[line_no].replace("#","").strip()[:9] and ' in ' in lines[line_no]:
			prelim_prob_stmt = "Parameters:\n" + "Parameters:".join("\n".join(lines[:line_no]).split("Parameters:")[1:]).strip()
			function_name = lines[line_no].split(" in ")[0].split("Function")[1].replace('"','').replace("'","").replace("`", "").strip()
			file_name = lines[line_no].split(" in ")[1].split(":")[0].replace('"','').replace("'","").replace("`", "").strip().split(" ")[-1]
			func_line = line_no
		elif "def " == lines[line_no][:4]:
			import_lines = "\n".join(lines[func_line+1:line_no]).strip().split("\n")
			function_def = "\n".join(lines[line_no:]).strip().replace("```python", "").replace("```", "")
			break

	prelim_prob_stmt = prelim_prob_stmt + "\n\n" + "The name of the function you create should be " + function_name
	
	final_function_def = []
	for fn_line in function_def.split("\n"):
		if len(fn_line) == 0:
			continue
		if len(fn_line) > 4 and "def " == fn_line[:4]:
			final_function_def.append(fn_line)
			continue
		elif len(fn_line.strip()) > 0 and fn_line[0] != " " and fn_line[0] != "\t" and fn_line[0] != ")":
			break
		final_function_def.append(fn_line)

	func_def = "\n".join(final_function_def)

	rel_funcs = []
	final_import_lines = []
	for imp_line in import_lines:
		if "import" in imp_line:
			final_import_lines.append(correct_import_line(imp_line))
		if "from " == imp_line.strip()[:5] and " import " in imp_line:
			cur_lib = imp_line[5:].split("import")[0].strip()
			std_flag = False
			for std_lib in standard_libs:
				if std_lib in cur_lib:
					std_flag = True
					break
			if not std_flag:
				rel_funcs.append((imp_line[5:].split("import")[1].strip(), imp_line[5:].split("import")[0].strip().split(".")[-1]))
	
	return prelim_prob_stmt, function_name, file_name, final_import_lines, func_def, rel_funcs


def remove_func_mentions(text, funcs):
	first_part = text.split("Objectives:")[0]
	objs_text = text.split("Objectives:")[1].split("Return Values:")[0]
	if "Return Values:" in text:
		last_part = text.split("Return Values:")[1]
	else:
		last_part = ""
	for func in funcs:
		objs_text = objs_text.replace(func["function_name"], "")
		# objs_text = objs_text.replace(func["function_name"].replace("_", " "), "")
		objs_text = objs_text.replace(func["file_name"], "")
		# objs_text = objs_text.replace(func["file_name"].replace(".py", ""), "")
	final_text = first_part + "Objectives:" + objs_text + "Return Values:" + last_part
	return final_text


def check_rel_func_test(test_code, rel_funcs):
	for func in rel_funcs:
		if "import " + func['function_name'] in test_code:
			return True
	return False


def get_called_functions(text):

	lines = text.split("\n")
	called_funcs = []
	for line_no in range(len(lines)):
		if "def " == lines[line_no][:4]:
			for temp_no in range(line_no+1, len(lines)):
				if "=" in lines[temp_no]:
					rhs = lines[temp_no].split("=")[1].strip()
					if len(rhs) > 0 and "(" in rhs and ")" in rhs:
						func_name = rhs.split("(")[0].strip()
						if " " not in func_name and "." not in func_name and "[" not in func_name:
							if func_name not in standard_funcs:
								called_funcs.append(func_name)
	return called_funcs
