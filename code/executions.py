import os
import traceback
import shutil
import pdb
import random
import subprocess

from prompts import get_generator_prompt
from utils import obtain_main_fn_txt, remove_python_wrapper


def get_function_exec_test(model, cur_function):
	main_fn_txt = obtain_main_fn_txt(cur_function)

	prompt, sys_prompt = get_generator_prompt("function_exec_check", params=main_fn_txt)

	og_pred = model.predict(prompt, sys_prompt, 2000, 0.5, 1, [])

	test_code = remove_python_wrapper(og_pred)
	test_code = test_code + "\n\nprint('All-Pass')"

	return test_code


# Check whether generated helper function executes correctly
def helper_exec_check(cur_function, test_code):
	sandbox_dir = "sandbox" + str(random.randint(0, 10000))
	if not os.path.exists(sandbox_dir):
		os.makedirs(sandbox_dir)

	with open("auxiliary/imports.txt", "r") as f:
		boiler_imports = f.read()

	with open(sandbox_dir + "/" + cur_function["file_name"], "w") as f:
		f.write(boiler_imports + "\n\n" + "\n".join(cur_function["import_lines"]) + "\n\n" + cur_function["function_def"])

	with open(sandbox_dir + "/test.py", "w") as f:
		f.write(boiler_imports + "\n\n" + test_code)

	exec_result = False

	try:
		result = subprocess.run(["python", sandbox_dir + "/test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
		output = result.stdout
		
		if result.returncode != 0:
			output += "\nError:\n" + result.stderr
		
		if output.strip().split("\n")[-1] == "All-Pass":
			exec_result = True
	except Exception as e:
		output = traceback.format_exc()

	shutil.rmtree(sandbox_dir)

	return exec_result, output


# Check whether generated problem executes correctly
def problem_exec_check(codebase_fns, main_fn, test_code):
	sandbox_dir = "sandbox" + str(random.randint(0, 10000))
	if not os.path.exists(sandbox_dir):
		os.makedirs(sandbox_dir)

	with open("auxiliary/imports.txt", "r") as f:
		boiler_imports = f.read()

	for fn in codebase_fns:
		if os.path.exists(sandbox_dir + "/" + fn["file_name"]):
			with open(sandbox_dir + "/" + fn["file_name"], "r") as f:
				cur_code = f.read()
			import_str = "\n".join(fn["import_lines"])
			new_code = boiler_imports + "\n\n" + import_str + "\n\n" + cur_code + "\n\n" + fn["function_def"]
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(new_code)
		else:
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(boiler_imports + "\n\n" + "\n".join(fn["import_lines"]) + "\n\n" + fn["function_def"])

	with open(sandbox_dir + "/" + main_fn["file_name"], "w") as f:
		f.write(boiler_imports + "\n\n" + "\n".join(main_fn["import_lines"]) + "\n\n" + main_fn["function_def"])

	with open(sandbox_dir + "/test.py", "w") as f:
		f.write(boiler_imports + "\n\n" + test_code)

	exec_result = False

	try:
		result = subprocess.run(["python", sandbox_dir + "/test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
		output = result.stdout
		
		if result.returncode != 0:
			output += "\nError:\n" + result.stderr
		
		if output.strip().split("\n")[-1] == "All-Pass":
			exec_result = True
	except Exception as e:
		output = traceback.format_exc()

	shutil.rmtree(sandbox_dir)

	return exec_result, output


# Check whether generated test code executes correctly for the problem
def execution(codebase_fns, main_fn, test_code):
	sandbox_dir = "sandbox" + str(random.randint(0, 10000))
	if not os.path.exists(sandbox_dir):
		os.makedirs(sandbox_dir)

	with open("auxiliary/imports.txt", "r") as f:
		boiler_imports = f.read()

	for fn in codebase_fns:
		if os.path.exists(sandbox_dir + "/" + fn["file_name"]):
			with open(sandbox_dir + "/" + fn["file_name"], "r") as f:
				cur_code = f.read()
			import_str = "\n".join(fn["import_lines"])
			new_code = boiler_imports + "\n\n" + import_str + "\n\n" + cur_code + "\n\n" + fn["function_def"]
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(new_code)
		else:
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(boiler_imports + "\n\n" + "\n".join(fn["import_lines"]) + "\n\n" + fn["function_def"])

	with open(sandbox_dir + "/" + main_fn["file_name"], "w") as f:
		f.write(boiler_imports + "\n\n" + "\n".join(main_fn["import_lines"]) + "\n\n" + main_fn["function_def"])

	test_code = boiler_imports + "\n\nfrom " + main_fn["file_name"].replace(".py", "") + " import " + main_fn["function_name"] + "\n\n" + test_code

	with open(sandbox_dir + "/test.py", "w") as f:
		f.write(test_code)

	exec_result = False

	try:
		result = subprocess.run(["python", sandbox_dir + "/test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
		output = result.stdout
		
		if result.returncode != 0:
			output += "\nError:\n" + result.stderr
		
		if output.strip().split("\n")[-1] == "All-Pass":
			exec_result = True
	except Exception as e:
		output = traceback.format_exc()

	shutil.rmtree(sandbox_dir)

	return exec_result, output


# Check whether model can solve the problem without context difficulties
def prediction_check(codebase_fns, prediction_code, answer_code, test_code):
	sandbox_dir = "sandbox" + str(random.randint(0, 10000))
	if not os.path.exists(sandbox_dir):
		os.makedirs(sandbox_dir)

	with open("auxiliary/imports.txt", "r") as f:
		boiler_imports = f.read()

	for fn in codebase_fns:
		if os.path.exists(sandbox_dir + "/" + fn["file_name"]):
			with open(sandbox_dir + "/" + fn["file_name"], "r") as f:
				cur_code = f.read()
			import_str = "\n".join(fn["import_lines"])
			new_code = boiler_imports + "\n\n" + import_str + "\n\n" + cur_code + "\n\n" + fn["function_def"]
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(new_code)
		else:
			with open(sandbox_dir + "/" + fn["file_name"], "w") as f:
				f.write(boiler_imports + "\n\n" + "\n".join(fn["import_lines"]) + "\n\n" + fn["function_def"])

	with open(sandbox_dir + "/" + answer_code["file_name"], "w") as f:
		f.write(boiler_imports + "\n\n" + prediction_code)

	test_code = boiler_imports + "\n\nfrom " + answer_code["file_name"].replace(".py", "") + " import " + answer_code["function_name"] + "\n\n" + test_code

	with open(sandbox_dir + "/test.py", "w") as f:
		f.write(test_code)
	
	exec_result = False

	try:
		result = subprocess.run(["python", sandbox_dir + "/test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
		output = result.stdout
		
		if result.returncode != 0:
			output += "\nError:\n" + result.stderr
		
		if output.strip().split("\n")[-1] == "All-Pass":
			exec_result = True
	except Exception as e:
		output = traceback.format_exc()

	shutil.rmtree(sandbox_dir)

	return exec_result, output


# Evaluation of the prediction
def evaluate_prediction(codebase, prediction_code, answer_code, test_code):
	sandbox_dir = "sandbox" + str(random.randint(0, 10000))
	if not os.path.exists(sandbox_dir):
		os.makedirs(sandbox_dir)

	with open("auxiliary/imports.txt", "r") as f:
		boiler_imports = f.read()

	for file_name in codebase:
		with open(sandbox_dir + "/" + file_name, "w") as f:
			all_imports = ""
			fn_defs = ""
			for fn in codebase[file_name]:
				import_str = "\n".join(fn["import_lines"])
				all_imports += import_str + "\n"
				fn_defs += fn["function_def"] + "\n\n"
			f.write(boiler_imports + "\n\n" + all_imports + "\n\n" + fn_defs)

	with open(sandbox_dir + "/" + answer_code["file_name"], "w") as f:
		f.write(boiler_imports + "\n\n" + prediction_code)

	test_code = boiler_imports + "\n\nfrom " + answer_code["file_name"].replace(".py", "") + " import " + answer_code["function_name"] + "\n\n" + test_code

	with open(sandbox_dir + "/test.py", "w") as f:
		f.write(test_code)
	
	exec_result = False

	try:
		result = subprocess.run(["python", sandbox_dir + "/test.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
		output = result.stdout
		
		if result.returncode != 0:
			output += "\nError:\n" + result.stderr
		
		if output.strip().split("\n")[-1] == "All-Pass":
			exec_result = True
	except Exception as e:
		output = traceback.format_exc()

	shutil.rmtree(sandbox_dir)

	return exec_result, output