def get_verification_prompt(prompt_type, params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "presence":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and an answer point. You must check two things:
1. Presence: Is the point mentioned in the document?
2. Relevance: Is the point discussed in a manner such that it can be used to partially answer the question?

Document:
{params[2]}

Question: {params[0]}

Answer Point:
{params[1]}

Give output in the following format:
Presence: True/False
Explanation for Presence:

Relevance: True/False
Explanation for Relevance:
"""

	elif prompt_type == "predict":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given a document followed by a question. You need to generate the answer for that question based on only the information provided in the document. Provide the answer in 1-3 bullet points, so that it is easier to objectively evaluate. Note that the document might not provide all the necessary information for answering the question, but you must just form the partial answer based on whatever information is given. Generate only the answer and nothing else. There is no need to mention what information is not provided in the document, just write the answer as best as you can. If the document does not provide any information relevant to the question, and there is no way you can attempt to generate even a partial answer, then you must only generate "No relevant information found in this document." and nothing else.

Document:
{params[1]}

Question: {params[0]}

Answer:"""

	elif prompt_type == "compare":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. You need to evaluate whether the prediction contains any major additional points that are not mentioned in the ground-truth answer. You can discard any points in the prediction that mention lack of information in the document about specific aspects. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of any new broad ideas or main points in the prediction that are not at all discussed in the ground-truth answer. The order of points mentioned is irrelevant. You should ignore the points in the prediction that just elaborate or provide more specifics or details over a major point in the ground-truth answer. You should also ignore if the ground-truth answer provides additional points that are not mentioned in the prediction.

If the prediction is not introducing major new points compared to the ground-truth answer, output "False" to "Presence of Extra Points" without giving any explanation. Otherwise, if and only if the prediction discusses major additional points that are not in the ground-truth answer, output "True" and mention only the extra major points part exactly as they are written in the prediction.

Question: {params[0]}

Ground-truth Answer: {params[1]}

Prediction: {params[2]}

Give output in the following format:
Presence of Extra Points: True/False
Extra Points Mentioned (if any):
"""

	elif prompt_type == "extra":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question and some answer points. You must check whether there are any additional major points in the document that provide relevant information for answering the question that are currently missing from the answer. Follow these instructions:
1. Do not look for exact phrases or explicit mentions since the answer can have points that are a paraphrase of the same broad information. 
2. It is ok if the document provides more specifics or details about the points already in the answer or if it discusses them in more depth by introducing related information so you can ignore that. 
3. Check if the document introduces a new "major" idea or point that is crucial for answering the question and is not at all mentioned in the answer and is not an extension of the existing points in the answer.
4. Your job is not to check if the question can be sufficiently answered. You should ignore if the document or answer points are missing any points that are needed in the answer to the question.

If the document is not introducing major new points pertaining to the answer, output "False" to "Presence of Extra Points" without giving any explanation. Otherwise, if and only if the document discusses major additional points that are necessary to answer the question, output "True" and mention only the extra major points discussed.

Document:
{params[2]}

Question: {params[0]}

Answer Points:
{params[1]}

Give output in the following format:
Presence of Extra Points: True/False
Extra Points Mentioned (if any):
"""

	elif prompt_type == "extra_adv":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a document followed by a question. You must check whether there is any information in the document that is relevant for answering the question. If not, output "False" to "Presence of Relevant Information" without giving any explanation. Otherwise, if and only if the document discusses information that is necessary to answer the question, output "True" and mention only that information, summarized as bullet points.

{params[1]}

Question: {params[0]}

Give output in the following format:
Presence of Relevant Information: True/False
Relevant Points (if any):
"""

	elif prompt_type == "remove":
		sys_prompt = "You are an expert at correcting data."
		prompt = f"""You are given a document followed by a question. Then you are provided with some information to be removed. Your job is to re-write the document exactly as it is, only removing those lines that refer to the information to be removed.

{params[3]}

Question: {params[0]}

Information to be removed:
{params[2]}"""

		if params[1] is not None:
			prompt = prompt + f"""

Note that the following points are crucial for answering the question and hence make sure that they must still be conveyed in the document:
{params[1]}"""
		
		prompt = prompt + f"""

Give output in the following format:
Title: <Title>
Text:
<Text>
"""

	elif prompt_type == "add":
		sys_prompt = "You are an expert at correcting data."
		prompt = f"""You are given a document followed by a question. Then you are provided with some information relevant for answering the question that must be added to the document. Your job is to re-write the document exactly as it is and placing this new relevant information somewhere inside. Do not elaborate or provide extra details or specifics surrounding this new information. Just directly mention it in a suitable way in context of the question.

{params[2]}

Question: {params[0]}

Information to be added:
{params[1]}

Give output in the following format:
Title: <Title>
Text:
<Text>
"""

	elif prompt_type == "adv_cross_check":
		sys_prompt = "You are an expert at verifying data."
		prompt = f"""You are given a question and an answer. You must check whether the answer is even partially relevant for answering the question. If the answer is not relevant at all, output "False" to "Relevance". Otherwise, if and only if the answer discusses information that is at least partially necessary to answer the question, output "True".

Question: {params[0]}

Answer:
{params[1]}

Give output in the following format:
Relevance: True/False
"""

	return prompt, sys_prompt



def get_evaluator_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "zero-shot-basic":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, an adversarial wrong answer, the ground-truth answer, and a prediction. You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. All the main points in the ground-truth answer must be mentioned in the prediction. The order of points mentioned is irrelevant. It is allowed for the prediction to elaborate or provide more specifics or details over the major points in the ground-truth answer. However, the prediction should not contain additional major points that are contradictory or irrelevant for answering the question. Importantly, the prediction must not discuss any of the points mentioned in the "adversarial wrong answer". The first word in your response must be either True or False. If False, explain why you think the prediction is wrong in detail.

Question: {question[0]}

Adversarial Wrong Answers: {question[3]}

Ground-truth Answer: {question[1]}

Prediction: {question[2]}

Result:
"""
	
	elif prompt_type == "zero-shot-step-by-step":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. 

Question: {question[0]}

Ground-truth Answer:
{question[1]}

Prediction:
{question[2]}

You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can have points that are a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. All the main points in the ground-truth answer must be mentioned in the prediction. The order of points mentioned is irrelevant. It is allowed for the prediction to elaborate or provide more specifics or details over the major points in the ground-truth answer. However, the prediction should not introduce additional major points that are contradictory or irrelevant for answering the question. Importantly, the prediction must not discuss the points in the "adversarial answer" and "adversarial information" provided below unless they are also mentioned in the Ground-truth answer.

Adversarial Answer:
{question[3]}

Adversarial Information:
{question[4]}

Do a step-by-step evaluation:
1. Check if all major points from the ground truth are covered in the prediction.
2. Check if the prediction introduces additional major points that are irrelavant or contradictory.
3. Check if the prediction mentions the points in the adversarial answer or adversarial information.

The last line of your response must state "FINAL VERDICT: Correct" if the evaluation passes or "FINAL VERDICT: Wrong" if the evaluation fails.
"""

	elif prompt_type == "zero-shot-score":
		sys_prompt = "You are an expert evaluator."
		prompt = f"""You are given a question, the ground-truth answer, and a prediction. You need to evaluate whether the prediction is correct by matching against the ground truth answer. Do not look for exact phrases or words since the prediction can be a paraphrase of the same information. Based on the question, check for the presence of the same ideas or main points in the prediction as in the ground-truth answer. It is okay if the prediction mentions any additional information. 
		You must assign a score of 10 if the prediction mentions all the information (even if paraphrased) present in the ground-truth answer. You must assign a score of 5 if the prediction covers most of the major points, but fails to mention 1-2 minor points present in the ground-truth. Otherwise you must give a score of 0.
		The first word in your response must be the score. Explain your score by saying what is missing from the prediction which is mentioned in the ground-truth answer.

Question: {question[0]}

Ground-truth Answer: {question[1]}

Prediction: {question[2]}

Result:
"""

	return prompt, sys_prompt




def get_solver_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "zero-shot-basic":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given some documents followed by a question. You need to generate the answer for that question. Provide the answer in bullet points, so that it is easier to objectively evaluate. Answering the question correctly requires information from multiple documents. If you find no relevant information in the documents for answering the question, you must only generate "No relevant information found in the documents." and nothing else.

Documents:
{question[0]}

Question: {question[1]}

Answer:
"""
	elif prompt_type == "zero-shot-no-irrelevant":
		sys_prompt = "You are an expert at answering questions based on documents."
		prompt = f"""You are given some documents followed by a question. You need to generate the answer for that question. Provide the answer in bullet points, so that it is easier to objectively evaluate. Answering the question correctly requires information from multiple documents. You must only generate the points necessary for answering the question, without mentioning anything irrelevant to the question. If you find no relevant information in the documents for answering the question, you must only generate "No relevant information found in the documents." and nothing else.

Documents:
{question[0]}

Question: {question[1]}

Answer:
"""

	return prompt, sys_prompt





def get_generator_prompt(prompt_type, question=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "programmatic_docs":
		sys_prompt = "You are an expert data generator. Following the instruction, you must generate long and correct documents."
		prompt = f"""You need to generate the documents for an example of a retrieval based Question Answering Task. 
The task consists of n documents provided in English text that consist of information about different topics and a question. To answer the question correctly compulsorily requires using some of the information in some subset of the documents provided.

Given below is a situation faced by {question[0]} when searching {question[1]}. The question-answer pair is:
Question: {question[2]}
Answer: {question[3]}

Given below are the assigned answer points for each document.
{question[4]}

Your job is to create long documents according to this information. For each document, first create 10-12 unique other points that are in no way related to the topic of the question and answer (different points for each document). These points should discuss very different things about a similar but different topic. Then use these points along with the assigned answer points to create a long document (at least 700 words long). The assigned answer points must be discussed taking into account the question. You must only discuss about these points and nothing else. Change the order of the points so that the answer points are embedded inside the document. Assign an appropriate title to the document. Do not summarize or conclude the document in the end.

Additionally, ensure that the documents you create do not have any information related to the following adversarial question-answer pairs. You should create documents that discuss topics that are completely different from the following information.
{question[-1]}

Give output in the following format:
Document 1:
Title: <Title>
Question: {question[2]}
Answer points assigned [Only these points must be covered with respect to the question]: <Points>
Other unrelated points created: <Points>
Text:
<Document Text>

Document 2:
Title: <Title>
Question: {question[2]}
Answer points assigned [Only these points must be covered with respect to the question]: <Points>
Other unrelated points created: <Points>
Text:
<Document Text>

and so on...
"""

	elif prompt_type == "programmatic_adversarial":
		sys_prompt = "You are an expert generator of data. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make hard data to test an advanced question answering system. You are given a question that a {question[0]} might want answered, along with the corresponding answer, and information of documents from {question[1]} that are important for answering that question. 

Original Question: {question[2]}

Original Answer:
{question[3]}

Original Documents:
{question[4]}

You must generate an adversarial question, adversarial answer, and corresponding adversarial documents that ask for something different but on similar topics or type so that it is difficult to answer the original question. Examples of how adversarial questions should look like are provided below:

Original Question: What are the best activities to do in Montreal, Canada during the winter season?
Adversarial Question: What activities should I look at when visiting Tokyo during the summer?

Original Question: What compliance steps are required to open a laundry business in Mumbai?
Adversarial Question: Provide the necessary steps to be taken to open an ice-cream shop in London.

Original Question: What are the circumstances when a court in Russia would re-open a divorce settlement case?
Adversarial Question: Provide the possibilities under which a judge in new york may warrant re-opening of a murder case?

Also provide an answer to the adversarial question, which is similar in style to the original answer, but differs significantly in information or specifics. The answer points for the adversarial question should be written in context of that adversarial question, so that they cannot be confused with the original question. Note that none of the points appearing in the original answer should be present in the answer to the adversarial question.

The answer to the adversarial question you craft must be scattered across different documents (at least 3) separate from the original answer documents. Assign each point of the adversarial answer to a specific document in which that point will be discussed. You may assign multiple points to the same adversarial document, but each point must only be assigned to a single adversarial document. You must state the title and adversarial answer points assigned for each of the adversarial documents. These adversarial documents should not have any overlapping information with the original answer documents."""
		if len(question[5]) > 0:
			prev_adv_questions = "\n\n".join(question[5])
			prompt = prompt + f"""

The following are adversarial questions I have already generated. Make a very different adversarial question.
{prev_adv_questions}"""
		prompt = prompt + f"""

Answer in the following format:

Question: <Question>
Answer: <Answer>

Document 1 Title: <Title>
Document 1 Answer points assigned: <Points>

Document 2 Title: <Title>
Document 2 Answer points assigned: <Points>

and so on...
"""
	
	elif prompt_type == "programmatic_qa":
		sys_prompt = "You are an expert generator of data. Do not use ** to start lines or denote points."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system.
Give me an example question and corresponding answer that a {question[0]} may ask that compulsorily requires searching a {question[1]}. Make questions that cannot be answered directly with general knowledge but necessarily require some uncommon information that is present in some documents. The answer must be very specific and written in bullet points, so that it is easier to objectively evaluate. Depending on the question, the answer can have anything between 3-6 bullet points without any sub-points.

The answer to the question you create must be scattered across different documents (at least 3). Assign each point of the answer to a specific document in which that point will be discussed. You may assign multiple points to the same document, but each point must only be assigned to a single document. You must state the title and answer points assigned for each of the documents.

Answer in the following format:

Question: <Question>
Answer: <Answer>

Document 1 Title: <Title>
Document 1 Answer points assigned: <Points>

Document 2 Title: <Title>
Document 2 Answer points assigned: <Points>

and so on...
"""

	elif prompt_type == "programmatic_scenarios":
		sys_prompt = "You are an expert generator of data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system.
Give me 5 examples of real-life scenarios where a USER_PERSONA may seek information in a COLLECTION_OF_DOCS. Do not consider educational or historical scenarios.

Some examples are:
{question}

Answer in the following format:
USER_PERSONA:
COLLECTION_OF_DOCS:
"""
	
	elif prompt_type == "naive_baseline":
		sys_prompt = "You are an expert generator of data."
		prompt = f"""You are a research scientist. You want to make data to test an advanced question answering system.
Given below is an example of a hard problem consisting of a set of documents, a question, and the corresponding answer. 

{question}

You need to generate a different problem for this same task. You must provide a different set of documents, a different question, and a different answer. The question must be such that it cannot be answered directly with general knowledge but necessarily requires some uncommon information that is present in the documents. The answer must be very specific and written in 3-4 bullet points, so that it is easier to objectively evaluate.

Answer in the following format:
Documents:
<Documents>

Question: <Question>

Answer:
<Answer>
"""

	return prompt, sys_prompt