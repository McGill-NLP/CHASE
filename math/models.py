import os
import time
import argparse
import uuid
from typing import List
import pdb

import openai
import anthropic
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.api_core.exceptions as gemini_exceptions
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.wait import wait_random_exponential
import tiktoken

from vllm import LLMEngine, EngineArgs, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

def load_model(model_name, peft_model=None, pp_size=1, tp_size=4):
	additional_configs = {}
	if peft_model:
		additional_configs["enable_lora"] = True
	else:
		additional_configs["enable_lora"] = False
		
	engine_config = EngineArgs(
		model=model_name,
		pipeline_parallel_size=pp_size,
		tensor_parallel_size=tp_size,
		max_loras=8,
		max_num_batched_tokens=65528,
		max_model_len=8000,
		**additional_configs)

	llm = LLMEngine.from_engine_args(engine_config)
	return llm

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_completion_response(client, engine, prompt, sys_prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty, best_of, logprobs=1, echo=False):
	fin_prompt = sys_prompt + "\n\n" + prompt
	return client.completions.create(
		model=engine,
		prompt=fin_prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty,
		best_of=best_of,
		logprobs=logprobs,
		echo=echo
	)

@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_chat_response(client, engine, prompt, sys_prompt, max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty):
	if "Mixtral-8x22B" in engine or "gemma" in engine:
		return client.chat.completions.create(
			model=engine,
			messages = [
				{"role": "user", "content": sys_prompt + "\n\n" + prompt}
			],
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			n=n,
			stop=stop,
			presence_penalty=presence_penalty,
			frequency_penalty=frequency_penalty
		)
	return client.chat.completions.create(
		model=engine,
		messages = [
			{"role": "system", "content": sys_prompt},
			{"role": "user", "content": prompt}
		],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		n=n,
		stop=stop,
		presence_penalty=presence_penalty,
		frequency_penalty=frequency_penalty
	)


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				openai.RateLimitError,
				openai.APIConnectionError,
				openai.InternalServerError,
				openai.APITimeoutError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_strawberry_response(client, engine, prompt):
	return client.chat.completions.create(
		model=engine,
		messages = [
			{"role": "user", "content": prompt}
		]
	)


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				gemini_exceptions.ResourceExhausted
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_gemini_response(model, prompt, max_tokens, temperature, top_p, n, stop):
	return model.generate_content(
		prompt,
		generation_config=genai.types.GenerationConfig(
			# Only one candidate for now.
			candidate_count=n,
			stop_sequences=stop,
			max_output_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
		),
		safety_settings={
			HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
		}
	)


@retry(
	retry=retry_if_exception_type(
			exception_types=(
				anthropic.RateLimitError,
				anthropic.InternalServerError,
				anthropic.APIConnectionError
			)
		),
		wait=wait_random_exponential(
			multiplier=0.1,
			max=0.5,
		),
	)
def _get_anthropic_response(client, engine, prompt, sys_prompt, max_tokens, temperature, top_p, stop):
	return client.messages.create(
		model=engine,
		system=sys_prompt,
		messages = [
			{"role": "user", "content": prompt}
		],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		stop_sequences=stop
	)


def peft_inference(
	model,
	peft_model_name=None,
	max_new_tokens=100,
	user_prompt=None,
	top_p=1.0,
	temperature=0.8,
	stop=[]
):
	sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens, stop=stop)

	lora_request = None
	if peft_model_name:
		lora_request = LoRARequest("lora",0,peft_model_name)

	req_id = str(uuid.uuid4())

	model.add_request(
		req_id,
		user_prompt,
		sampling_param,
		lora_request=lora_request
	)

	while model.has_unfinished_requests():
		request_outputs: List[RequestOutput] = model.step()

		if request_outputs[-1].finished:
			return request_outputs[-1].outputs[0].text

	return "LOLZ"


class LargeLanguageModel():
	def __init__(self, model_type, model, peft_model, sys_prompt, top_p, presence_penalty, frequency_penalty, port=8080, timeout=1000000):
		self.model_type = model_type
		self.engine = model
		self.peft_model = peft_model
		self.top_p = top_p
		self.presence_penalty = presence_penalty
		self.frequency_penalty = frequency_penalty

		if self.peft_model == "none":
			self.peft_model = None

		if self.model_type in ['vllm']:
			openai_api_key = "EMPTY"
			openai_api_base = "http://localhost:8000/v1"
			self.client = OpenAI(
				api_key=openai_api_key,
				base_url=openai_api_base,
			)
		elif self.model_type in ['chat', 'completion', 'o1']:
			self.client = OpenAI(
				api_key=openai.api_key,
			)
		elif self.model_type in ['anthropic']:
			self.client = anthropic.Anthropic(
				api_key=anthropic.api_key,
			)
		elif self.model_type in ['gemini']:
			self.gemini = genai.GenerativeModel(model_name=model, system_instruction=sys_prompt)
		elif self.model_type in ['peft']:
			self.llm = load_model(model_name=model, peft_model=peft_model)

	def predict(self, prompt, sys_prompt, max_tokens, temperature=0.0, n=1, stop = []):
		if self.model_type == "completion":
			response = _get_completion_response(
				client=self.client,
				engine=self.engine,
				prompt=prompt,
				sys_prompt=sys_prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				top_p=self.top_p,
				n=n,
				stop=stop,
				presence_penalty=self.presence_penalty,
				frequency_penalty=self.frequency_penalty,
				best_of=n+1,
				echo=False
			)
			response = response["choices"][0]['text'].lstrip('\n').rstrip('\n')
		elif self.model_type in ["chat", "vllm"]:
			# pdb.set_trace()
			response = "error"
			cur_max_tokens = max_tokens
			while(cur_max_tokens > 0):
				try:
					response = _get_chat_response(
						client=self.client,
						engine=self.engine,
						prompt=prompt, 
						sys_prompt=sys_prompt,
						max_tokens=cur_max_tokens,
						temperature=temperature,
						top_p=self.top_p,
						n=n,
						stop=stop,
						presence_penalty=self.presence_penalty,
						frequency_penalty=self.frequency_penalty
					)
					response = response.choices[0].message.content.lstrip('\n').rstrip('\n')
					break
				except openai.BadRequestError:
					cur_max_tokens = cur_max_tokens - 2000
		elif self.model_type in ["o1"]:
			response = _get_strawberry_response(
				client=self.client,
				engine=self.engine,
				prompt=prompt
			)
			response = response.choices[0].message.content.lstrip('\n').rstrip('\n')
		elif self.model_type in ["anthropic"]:
			response = _get_anthropic_response(
				client=self.client,
				engine=self.engine,
				prompt=prompt, 
				sys_prompt=sys_prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				top_p=self.top_p,
				stop=stop
			)
			response = response.content[0].text.lstrip('\n').rstrip('\n')
		elif self.model_type in ["gemini"]:
			response = _get_gemini_response(
				model=self.gemini,
				prompt=prompt,
				max_tokens=max_tokens,
				temperature=temperature,
				top_p=self.top_p,
				n=n,
				stop=stop
			)
			response = response.text.lstrip('\n').rstrip('\n')
		elif self.model_type in ["peft"]:
			response = peft_inference(
				model=self.llm,
				peft_model_name=self.peft_model,
				max_new_tokens=max_tokens,
				user_prompt=prompt,
				top_p=1.0,
				temperature=temperature,
				stop=stop
			)
			response = response.lstrip('\n').rstrip('\n')
		return response