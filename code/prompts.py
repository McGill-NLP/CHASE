def get_solver_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "basic":
		sys_prompt = "You are an expert programmer. You must output only python code."
		prompt = f"""You are given a codebase. You need to write a single python function to achieve the objectives specified in the problem statement. In your function, you should call some of the functions in the codebase to achieve specific objectives. Do not give any examples of usage or any explanations.

Codebase:

{params[0]}

Problem Statement:

{params[1]}

Give output in the following format:
```Python
# Import statements if required
import <>
...

# Import necessary helper functions from their files
from <filename> import <function_name>
...

# Define the function
def <function_name>(<param1>, <param2>, ...):
	# Your code here
	...

	return <return_variable>
```
"""
	
	return prompt, sys_prompt


def get_generator_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "function_exec_check":
		sys_prompt = "You are an expert tester of code systems."
		prompt = f"""You are given a function in a file. You need to check whether the function correctly executes.

{params}

Follow these instructions:
1. You must output only a single long python code. 
2. First initialize the input parameters for the function in python code. If the function reads data from files, you should create and store the necessary files with sample data in the corresponding filepath in the python code.
3. Finally, call the function with the input parameters.

Give output in the following format:
Input Parameters:
```Python
# Import statements if required
import <>
...

# Import function from file
from <filename> import <function_name>

# Initialize input parameters
<param1> = <value1>
<param2> = <value2>
...

# Call function with input parameters
return_<variable1>, return_<variable2>, ... = <function_name>(<param1>, <param2>, ...)
```
"""
	
	elif prompt_type == "test":
		sys_prompt = "You are an expert tester of code systems."
		prompt = f"""You are given a codebase and a function. You need to define an input-output test case for that function to exhaustively test all scenarios.

Codebase:

{params[0]}

Below is the main function you need to write input-output test cases for:

{params[1]}

Follow these instructions:
1. You must output only a single long python code. 
2. First initialize the input parameters for the function in python code. If the function reads data from files, you should create and store the necessary files with sample data in the corresponding filepath in the python code. Call the function and assign the return values to variables named as return_<variable_name>.
3. Then write new code to implement the exact logic of the function without using the other functions in the codebase. This way, you need to simulate step-by-step how the values of the input parameters will be used to obtain the final return values. Call these values as correct_<variable_name>.
4. Finally, and most importantly use assert statements to compulsorily check if the returned outputs of the function (return_<variable_name> variables) match with the ones you computed yourself (correct_<variable_name> variables).

Give output in the following format:
Input Parameters:
```Python
# Import statements if required
import <>
...

# Import function from file
from <filename> import <function_name>

# Initialize input parameters
<param1> = <value1>
<param2> = <value2>
...

# Call function with input parameters
return_<variable1>, return_<variable2>, ... = <function_name>(<param1>, <param2>, ...)

# Step-by-step run-through of function to obtain intermediate outputs:

# Step 1
# Explanation: <>
<Code for step-1>

# Step 2
# Explanation: <>
<Code for step-2>
...

# Final Expected Output:
correct_<variable1> = <value1>
correct_<variable2> = <value2>
...

# Assert statements (compulsory) to check if the function returns the correct values:
assert return_<variable1> == correct_<variable1>
assert return_<variable2> == correct_<variable2>
...
```
"""
	
	elif prompt_type == "verify_problems":
		sys_prompt = "You are an expert programmer."
		prompt = f"""You are given a codebase with some files and functions in the domain of {params[0]}. You need to write a single python function to achieve the objectives specified in the problem statement. You may call the functions in the codebase when necessary. Do not give any examples of usage or any explanations.

Codebase:

{params[1]}

Problem Statement:

{params[2]}

Give output in the following format:
```Python
# Import statements if required
import <>
...

# Import necessary helper functions from their files
from <filename> import <function_name>
...

# Define the function
def <function_name>(<param1>, <param2>, ...):
	# Your code here
	...

	return <return_variable>
```
"""
	
	elif prompt_type == "modify_problems":
		sys_prompt = "You are an expert data generator."
		prompt = f"""You are given a ground-truth code function along with a problem statement. However, when the problem statement was provided to an expert programmer, it resulted in a different and incorrect predicted code. Make minimal changes to the problem statement to make it more aligned with the ground-truth code so that the programmer won't generate the wrong code with it. In the problem statement, you must never mention the names of the helper functions that need to be called. You should not remove details or specifics from the problem statement and don't try to make it shorter.

Ground-truth Code:

{params[2]}

Problem Statement:

{params[0]}

Incorrect Predicted Code:

{params[1]}

Output only the modified problem statement and nothing else."""

	elif prompt_type == "helper_functions":
		sys_prompt = "You are an expert generator of code data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced code generation system. 

You are given a domain. Assume that there is a large python code base 'C' with at least 10 python files on that domain. 

Domain: {params[0]}

You need to create 5 functions in this codebase for achieving various objectives. 

First define the parameters that will be input to the function.

Then define the objective of the function. The objective must consist of 3-4 sub-goals, each of which must involve complex logic that make it very difficult to implement the function. However, each sub-goal must be well-specified such that there is only one way to implement the sub-goal. Then based on the objective, you need to create a single function (do not create other functions inside this).

Some examples are:

{params[1]}

Now you need to create 5 unique, diverse, and complex functions. Answer in the following format:

Function <Number>:
Parameters:
- <para_name>: <data_type>
- <para_name>: <data_type>
...
Objectives:
- <sub_goal>
- <sub_goal>
...

Function "function_name" in file "file_name.py":
<import statements>

<function definition>
"""

	elif prompt_type == "problem_statement":
		sys_prompt = "You are an expert generator of code data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced code generation system.

Below, you are given 10 functions from a codebase "C" in the domain of {params[0]}. 

{params[1]}

You need to create a complex function that calls at least 4 (but not more than 6) of these functions to achieve various objectives. Apart from just calling these functions, it should also implement some other pieces of complex logic. You first need to define the parameters that will be input to the function. Then you need to define the objective of the function. Follow these instructions for creating the objective:
1. The objective must consist of 6-8 sub-goals. Each sub-goal must be detailed and well-specified such that there is only one way to implement the sub-goal.
2. VERY IMPORTANT: The objective must not explicitly specify which functions should be called.
3. Always give names for variables you are talking about in the objective.
4. You must explicitly mention what parameters are to be used for a specific sub-goal by the name of the parameter.
5. Whenever a variable is obtained that must be returned by the function, you must explicitly state that in the sub-goal.
6. At least 2 of the sub-goals must involve some complex logic, apart from just calling helper functions that make it very difficult to implement the function.

Once you write down the objective, you need to create the function that achieves this objective. Import the required functions from the codebase "C" and use them in your function.

You must give output in the following format:

Parameters:
- <para_name>: <data_type> # Explanation of the parameter
- <para_name>: <data_type> # Explanation of the parameter
...
Objectives:
- <sub_goal>
- <sub_goal>
...
Return Values:
- <return_name>: <data_type> # Explanation of the return value
- <return_name>: <data_type> # Explanation of the return value
...

Function <Function_name> in <Name_of_file>:

<import statements only>

<function definition only>
"""
		
	elif prompt_type == "problem_statement_small":
		sys_prompt = "You are an expert generator of code data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced code generation system.

Below, you are given 10 functions from a codebase "C" in the domain of {params[0]}. 

{params[1]}

You need to create a complex function that calls at least 3 (but not more than 5) of these functions to achieve various objectives. Apart from just calling these functions, it should also implement some other pieces of complex logic. You first need to define the parameters that will be input to the function. Then you need to define the objective of the function. Follow these instructions for creating the objective:
1. The objective must consist of 3-5 sub-goals. Each sub-goal must be detailed and well-specified such that there is only one way to implement the sub-goal.
2. VERY IMPORTANT: The objective must not explicitly specify which functions should be called.
3. Always give names for variables you are talking about in the objective.
4. You must explicitly mention what parameters are to be used for a specific sub-goal by the name of the parameter.
5. Whenever a variable is obtained that must be returned by the function, you must explicitly state that in the sub-goal.
6. At least 2 of the sub-goals must involve some complex logic, apart from just calling helper functions that make it very difficult to implement the function.

Once you write down the objective, you need to create the function that achieves this objective. Import the required functions from the codebase "C" and use them in your function.

You must give output in the following format:

Parameters:
- <para_name>: <data_type> # Explanation of the parameter
- <para_name>: <data_type> # Explanation of the parameter
...
Objectives:
- <sub_goal>
- <sub_goal>
...
Return Values:
- <return_name>: <data_type> # Explanation of the return value
- <return_name>: <data_type> # Explanation of the return value
...

Function <Function_name> in <Name_of_file>:

<import statements only>

<function definition only>
"""

	return prompt, sys_prompt