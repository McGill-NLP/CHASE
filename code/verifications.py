import ast
import pdb

def para_return_check(fn_def, problem):
	para_cnt = 0
	return_cnt = 0
	lines = problem.split("\n")
	for line_no in range(len(lines)):
		if "Parameters:" in lines[line_no]:
			params = {}
			for temp_no in range(line_no+1, len(lines)):
				if "Objectives:" in lines[temp_no]:
					break
				elif "-" in lines[temp_no]:
					para_cnt += 1
					para_line = lines[temp_no].split("-")[1].strip().split(":")
					params[para_line[0].strip()] = para_line[1].strip()
			break
	for line_no in range(len(lines)):
		if "Return Values:" in lines[line_no]:
			returns = {}
			for temp_no in range(line_no+1, len(lines)):
				if "-" in lines[temp_no]:
					return_cnt += 1
					return_line = lines[temp_no].split("-")[1].strip().split(":")
					returns[return_line[0].strip()] = return_line[1].strip()

	if ")" in fn_def.split("\n")[0]:
		fn_paras = fn_def.split("\n")[0].split("(")[1].split(")")[0].split(",")
	else:
		fn_paras = []
		if "," in fn_def.split("\n")[0]:
			fn_paras = fn_def.split("\n")[0].split("(")[1].split(",")
		for line_no in range(1, len(fn_def.split("\n"))):
			if ")" in fn_def.split("\n")[line_no]:
				if len(fn_def.split("\n")[line_no].strip()) > 2:
					fn_paras.extend(fn_def.split("\n")[line_no].split(")")[0].split(","))
				break
			fn_paras.extend(fn_def.split("\n")[line_no].split(","))

	act_fn_para_cnt = 0
	skip = 0
	for fn_para in fn_paras:
		if skip > 0:
			if "[" in fn_para:
				skip = skip + fn_para.count("[")
			if "]" in fn_para:
				skip = skip - fn_para.count("]")
			continue
		if len(fn_para.strip()) > 0:
			act_fn_para_cnt += 1
			if "[" in fn_para:
				skip = fn_para.count("[") - fn_para.count("]")

	if para_cnt != act_fn_para_cnt:
		print("Para Checks failed...")
		print(para_cnt, act_fn_para_cnt)
		act_fn_para_cnt = 0
		return False
	
	act_fn_para_cnt = 0
	
	fn_returns = []
	for fn_line in fn_def.split("\n"):
		if "return " in fn_line:
			fn_returns = fn_line.strip()[7:].split(",")

	if return_cnt != len(fn_returns):
		print("Return Checks failed...")
		print(return_cnt, len(fn_returns))
		return False
	
	return True


def assert_check(test_code):
	test_lines = test_code.split("\n")
	assert_check = False
	for line in test_lines[-5:]:
		if len(line) > 0 and "#" != line.strip()[0] and "assert" in line and "None" not in line and ".shape" not in line:
			assert_check = True
			break
	return assert_check


class AssertAnalyzer(ast.NodeVisitor):
	def __init__(self, function_name):
		self.function_name = function_name
		self.func_returns = set()  # Variables assigned from function_name
		self.asserts = []          # Store asserts involving function returns
		self.var_assignments = {}  # Keep track of variable assignments

	def visit_Call(self, node):
		# Check if it's a call to the target function (function_name)
		if isinstance(node.func, ast.Name) and node.func.id == self.function_name:
			# Check if the function's return is assigned to a variable
			parent = node.parent
			if isinstance(parent, ast.Assign):
				for target in parent.targets:
					# Handle multiple return values (e.g., var1, var2 = function())
					if isinstance(target, ast.Tuple):
						for elt in target.elts:
							if isinstance(elt, ast.Name):
								self.func_returns.add(elt.id)
					# Single return value (e.g., var1 = function())
					elif isinstance(target, ast.Name):
						self.func_returns.add(target.id)

		# Check for pd.testing.assert_frame_equal or similar assert calls
		elif isinstance(node.func, ast.Attribute):
			# Check if the function is 'assert_frame_equal' or similar
			if node.func.attr == 'assert_frame_equal':
				# Check if any argument is a variable returned from function_name
				for arg in node.args:
					if isinstance(arg, ast.Name) and arg.id in self.func_returns:
						self.asserts.append(ast.dump(node))
		self.generic_visit(node)

	def visit_Assign(self, node):
		# Track all variable assignments
		if isinstance(node.targets[0], ast.Name):
			self.var_assignments[node.targets[0].id] = ast.dump(node.value)
		self.generic_visit(node)

	def visit_Assert(self, node):
		# Handle assert statements with method calls like equals()
		if isinstance(node.test, ast.Call):
			# Check if the method call is '.equals()'
			if isinstance(node.test.func, ast.Attribute) and node.test.func.attr == 'equals':
				# Check if the object calling equals() is a variable storing a function return
				if isinstance(node.test.func.value, ast.Name) and node.test.func.value.id in self.func_returns:
					self.asserts.append(ast.dump(node))
				# Check if the argument of equals() is a variable storing a function return
				for arg in node.test.args:
					if isinstance(arg, ast.Name) and arg.id in self.func_returns:
						self.asserts.append(ast.dump(node))
		# Handle assert statements with direct comparisons (e.g., assert return_df == correct_df)
		elif isinstance(node.test, ast.Compare):
			left_side = node.test.left
			right_side = node.test.comparators
			# Check if either side of the comparison is a variable that stores the function return
			if isinstance(left_side, ast.Name) and left_side.id in self.func_returns:
				self.asserts.append(ast.dump(node))
			for comparator in right_side:
				if isinstance(comparator, ast.Name) and comparator.id in self.func_returns:
					self.asserts.append(ast.dump(node))
		self.generic_visit(node)

	def generic_visit(self, node):
		# Update the parent reference for each node to allow checking assignments
		for child in ast.iter_child_nodes(node):
			child.parent = node
		super().generic_visit(node)


def verify_test_code(code, function_name):
	# Parse the code into an AST
	try:
		tree = ast.parse(code)
		analyzer = AssertAnalyzer(function_name)
		analyzer.visit(tree)
	except:
		return 0

	ver = 0
	if len(analyzer.asserts) > 0:
		ver = 1
	return ver