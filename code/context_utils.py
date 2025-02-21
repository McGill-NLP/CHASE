import json
import random

from utils import jaccard_similarity


def get_context_text(files):
	context = ""
	for file_name in files:
		context = context + "File: " + file_name + ":\n\n"
		import_lines = []
		defs = ""
		for func in files[file_name]:
			cur_import_lines = func["import_lines"]
			for imp_line in cur_import_lines:
				if imp_line not in import_lines:
					import_lines.append(imp_line)
			
			defs = defs + func["function_def"] + "\n\n"
		context = context + "```Python\n" + "\n".join(import_lines) + "\n\n" + defs.strip() + "\n```\n\n"
	
	return context.strip()


def sample_context(prompt_fns, domain, extra_fn = 10):
	with open("helper_functions/generated/" + domain + ".json", "r") as f:
		data = json.load(f)

	all_fns = random.sample(data, len(data))

	files = {}

	for func in prompt_fns:
		if func["file_name"] in files:
			files[func["file_name"]].append(func)
		else:
			files[func["file_name"]] = [func]

	start_idx = 0

	new_files = {}

	for file_name in files:
		cur_list = []
		for func in files[file_name]:
			cur_list.append(func)
			other_fns_sampled = extra_fn
			other_funcs = []
			
			cnt = 0
			for i in range(start_idx, len(all_fns)):
				cur_function = all_fns[i]
				sim_check = False
				for func in prompt_fns:
					name1 = cur_function["function_name"].split("_")
					filename1 = cur_function["file_name"].replace(".py","").split("_")
					name2 = func["function_name"].split("_")
					filename2 = func["file_name"].replace(".py","").split("_")
					if jaccard_similarity(set(name1 + filename1), set(name2 + filename2)) > 20:
						obj1 = set("\n".join(cur_function["objectives"]).split())
						obj2 = set("\n".join(func["objectives"]).split())
						if jaccard_similarity(obj1, obj2) > 5:
							sim_check = True
							break
				if not sim_check:
					other_funcs.append(cur_function)
					cnt += 1
					if cnt == other_fns_sampled:
						start_idx = i + 1
						break

			cur_list.extend(other_funcs)
		
		random.shuffle(cur_list)
		new_files[file_name] = cur_list
	
	context = get_context_text(new_files)

	return new_files, context