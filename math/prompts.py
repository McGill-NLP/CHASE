def get_solver_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "8-shot-cot":
		sys_prompt = "You are an expert mathematician. Your final statement must be of the form 'The answer is <answer>'."
		prompt = f"""Solve the final math word problem given below by thinking step-by-step. You should always work with exact numbers - never round down or round up decimals based on context. Give the final answer in the end by saying "The answer is <number>".

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {params}
A:"""
	
	elif prompt_type == "8-shot-cot-no-sys":
		sys_prompt = ""
		prompt = f"""Solve the final math word problem given below by thinking step-by-step. You should always work with exact numbers - never round down or round up decimals based on context. Give the final answer in the end by saying "The answer is <number>".

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
Q: {params}
A:"""
	
	elif prompt_type == "3-shot-sentence":
		sys_prompt = "You are an expert mathematician. Your final statement must be of the form 'The answer is <answer>'."
		prompt = f"""You need to solve the given math word problem. You should break down the problem sentence by sentence, and solve each sentence, one at a time, from start to finish until you get the final answer. You should always work with exact numbers - never round down or round up decimals based on context. Give the final answer in the end by saying "The answer is <number>".

Given below are illustrations of solving sentence-by-sentence:

Q: In a store, an Uno Giant Family Card costs $12. When Ivan bought ten pieces, he was given a discount of $2 for each. The store has a 8% sales tax added to all purchases. Ivan decides to save 25% of this expenditure for a future vacation. After saving, Ivan instead decides to split this amount between two of his friends who are always helping him out. One of the friends decided to split their received amount equally among the five children in Ivan's neighborhood who helped him move the previous day. How much will each child receive?
A: Sentence 1: Uno card costs $12.
Sentence 2: Ivan bought 10 cards and there was a discount of $2 each. So, 10 * $12 = $120 total cost of cards and, 10 * $2 = $20 discount.
Sentence 3: Sales tax is 8% of ($120 - $20 = $100). So, 8% of $100 = $8.
Sentence 4: Ivan saves 25% of ($100 + $8) = $108. So, 25% of $108 = $27.
Sentence 5: Ivan splits $27 between 2 friends. So, $27 / 2 = $13.50 each.
Sentence 6: One friend splits $13.50 among 5 children. So, $13.50 / 5 = $2.70 each.
The answer is 2.70.

Q: Carly is trying to get in shape to try out for the soccer team. She starts by running 2 miles a week. The second week, she runs twice as long plus 3 extra miles per week. The third week she runs 9/7 as much as she ran the second week. The week after that, she sprains her ankle and has to reduce her running time by 5 miles this week compared to the previous week. After a few weeks of recovering from her ankle injury, Carly starts to feel better and decides to gradually increase her running time. She starts with a shorter routine that is one quarter of the amount she ran the week she was injured. After a week of intense training, Carly decides to boost her speed and endurance by increasing her weekly running routine to 2.5 times longer. Carly is planning increase the weekly routine by 5 times now. How much is Carly planning to run every week?
A: Sentence 1: Nothing to solve.
Sentence 2: Carly runs 2 miles a week.
Sentence 3: Carly runs 2 * 2 + 3 = 7 miles in the second week.
Sentence 4: Carly runs 9/7 * 7 = 9 miles in the third week.
Sentence 5: Carly reduces her running time by 5 miles this week. So, 9 - 5 = 4 miles.
Sentence 6: Nothing to solve.
Sentence 7: Carly starts with 1/4 of 4 miles = 1 mile.
Sentence 8: Carly increases her running routine to 2.5 times longer. So, 1 * 2.5 = 2.5 miles.
Sentence 9: Carly wants to make her long run 5 times as long as her new routine. So, 2.5 * 5 = 12.5 miles.
The answer is 12.5.

Q: Mark needs to replace the radiator in his car. The work takes 2 hours and the mechanic charges $75 an hour. The part cost $150. Mark wants to treat the mechanic for a good job, so he decides to give him a 10% tip on the total amount he paid. The owner of the service company decides to offer the same amount as Mark's generous tip split evenly to two other mechanics. A mechanic distributed the money they received to 5 different charities. If 3 of the charities pooled together to buy clothes costing $0.5 each using the donated money, how many clothes did the charities buy in total?
A: Sentence 1: Nothing to solve.
Sentence 2: The work takes 2 hours and the mechanic charges $75 an hour. So, 2 * $75 = $150.
Sentence 3: The part cost $150. So, $150 + $150 = $300 is the total cost so far.
Sentence 4: Mark gives a 10% tip on $300. So, 10% of $300 = $30.
Sentence 5: The owner splits $30 between 2 mechanics. So, $30 / 2 = $15 each.
Sentence 6: The mechanic distributes $15 to 5 charities. So, $15 / 5 = $3 each.
Sentence 7: 3 charities pool together to buy clothes costing $0.5 each. So, (3 * 3) / $0.5 = 18.
The answer is 18.

Similarly, solve the following question:

Q: {params}
A:"""
	
	return prompt, sys_prompt


def get_verification_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "check_banned_info":
		sys_prompt = "You are an expert at verifying data. Your final statement must start with 'yes' or 'no'."
		prompt = f"""You are given a math word problem and some adversarial information. You must check whether any of the facts provided in the adversarial information is mentioned or contradicted in the problem. 

Do this step by step for each fact in the adversarial information. If you find a fact that is (1) mentioned or (2) contradicted, you must say 'yes' and explain how. If you found any fact that was mentioned or contradicted, then in your final statement, you must say "yes, facts were mentioned or contradicted". Otherwise, if none of the facts have been mentioned or contradicted, say "no, facts were not mentioned or contradicted" in the final statement.

Question: {params[0]}
Adversarial Information: {params[1]}"""

	elif prompt_type == "grammar_correct":
		sys_prompt = "You are an expert English and Mathematics teacher."
		prompt = f"""You are given three pieces of a math word problem's context. Your task is to correct the grammar, punctuation and language of all the three pieces. Each context should consist of properly ended sentences, and should not pose any questions. Make sure you do not change any important information or fact, and that you do not add any new information. Also make sure that the three pieces of context flow easily and logically from one to the next.

Context 1: {params[0]}

Context 2: {params[1]}

Context 3: {params[2]}

Give output in the following format only:
Corrected context 1: <>

Corrected context 2: <>

Corrected context 3: <>"""
	
	return prompt, sys_prompt


def get_generator_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "problem_extend":
		sys_prompt = "You are an expert mathematician."
		if params is None:
			return "", sys_prompt
		prompt = f"""You are a research scientist. Your task is to create a hard math word problem to test an advanced math reasoning system. For that, you are given the following problem:

Q: {params[0]}
A: {params[1]}

Your job is to first divide up the problem into the "context" and the "question statement". Isolate the quantity that the problem is inquiring about by looking at the final question statement and the rest of the information provided becomes the context. Also form a brief answer statement by phrasing the answer in a complete sentence. Do not include the answer statement in the context.

Then you need to further continue the problem over that quantity, by introducing a scenario and final question where you need to perform one more operation (such as +,-,/,*, etc.) over this quantity to get the final answer. Crucially, the new context must not mention the original answer - it still has to be inferred based on previous information. Do not make any calculation or inference in the new context. Try to make the new context challenging. Also provide a complete reasoning of how you reached the new answer (never round down or round up decimals).

Give output in the following format only:
Original context [without question statement]: <>
Question statement: <>
Original answer: <>
Original answer statement: <>

New operation over original answer: <>
New context [Do not mention original answer]: <>
New question statement: <>
New answer reasoning: <>
New answer [Number only]: <>"""
	
	elif prompt_type == "problem_extend_small_model":
		sys_prompt = "You are an expert mathematician. Do not output <>. Do not make assumptions in your reasoning. Do not output any seed question or seed answer. Limit your new answer reasoning to 2-3 sentences. End your output with the new answer in numerical form (without calculation or equation)."
		if params is None:
			return "", sys_prompt
		prompt = f"""You are a research scientist. Your task is to create a hard math word problem to test an advanced math reasoning system. For that, you are given the following problem:

Q: {params[0]}
A: {params[1]}

Your job is to first divide up the problem into the "context" and the "question statement". Isolate the quantity that the problem is inquiring about by looking at the final question statement and the rest of the information provided becomes the context. Also form a brief answer statement by phrasing the answer in a complete sentence. Do not include the answer statement in the context.

Then you need to further continue the problem over that quantity, by introducing a scenario and final question where you need to perform one more operation (such as +,-,/,*, etc.) over this quantity to get the final answer. Do not use any other information from the original question, you just have to assume the information in the original answer statement and build on that. Crucially, the new context must not mention the original answer - it still has to be inferred based on previous information. Do not make any calculation or inference in the new context. Also provide a complete reasoning of how you reached the new answer (never round down or round up decimals).

Give output in the following format only:
Original context: <>
Question statement: <>
Original answer: <>
Original answer statement: <>

New operation over original answer: <>
New context [Do not mention original answer]: <>
New question statement: <>
New answer reasoning: <>
New answer: <>"""
	
	elif prompt_type == "naive_baseline":
		sys_prompt = "You are an expert mathematician. If possible, avoid creating problems requiring calculations with decimal points. You must provide final numerical answer in the form of #### <number> in the end."
		prompt = f"""I want you to act as a problem rewriter.
Your objective is to rewrite a given math word problem into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle. But the rewritten problem must be correct, reasonable and must be understood and responded by humans. You SHOULD complicate the given problem using the following method:
If given problem can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.

Given problem: {params[0]}
Given answer: {params[1]}

Give output in the following format only:
Rewritten problem: <>
New answer: <>
"""

	return prompt, sys_prompt