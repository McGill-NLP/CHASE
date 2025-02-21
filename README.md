<h2 align="center">
  CHASE: Challenging AI with Synthetic Evaluations
</h2>
<!-- <h5 align="center">Evaluating the In-Context Learning Ability of Large Language Models to Generalize to Novel Interpretations</h5> -->

<p align="center">
  <a href="https://paper"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://github.com/McGill-NLP/incontext-code-generation/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

<p style="text-align: justify;">
The pace of evolution of Large Language Models (LLMs) necessitates new approaches for rigorous and comprehensive evaluation. Traditional human annotation is increasingly impracticable due to the complexities and costs involved in generating high-quality, challenging problems. In this work, we introduce **CHASE**, a unified framework to synthetically generate challenging problems using LLMs without human involvement.  For a given task, our approach builds a hard problem in a bottom-up manner from simpler components. Moreover, our framework decomposes the generation process into independently verifiable sub-tasks, thereby ensuring a high level of quality and correctness. We implement CHASE to create evaluation benchmarks across three diverse domains: (1) document-based question answering, (2) repository-level code completion, and (3) math reasoning. The performance of state-of-the-art LLMs on these synthetic benchmarks lies in the range of 40-60% accuracy, thereby demonstrating the effectiveness of our framework at generating challenging problems.
</p>
<h2 align="center">
  <img align="center"  src="./images/Fig_main.svg" alt="..." width="800">
</h2>



# Setup

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 chasenv
$ source chasenv/bin/activate
```

Depending on your machine, you may have to do:

```shell
$ python3 -m venv chasenv
$ source chasenv/bin/activate
```

## Dependencies

- compatible with python 3
- dependencies can be installed using `CHASE/requirements.txt`
- Works best with CUDA 12.5 (otherwise you may have to struggle with installation of individual libraries)

Install all the required packages:

at `CHASE/:`

```shell
$ pip install -r requirements.txt
```

# Usage

Here, we illustrate running the experiments for generation and solving using GPT-4o-mini, Gemini-1.5-Flash and Llama-3.3-70B.

## IA. Generating CHASE-QA

### Generate scenarios

We shall use GPT-4o-mini to generate 50 scenarios (each iteration generates 5). Note that some scenarios may get rejected because of high similarity with existing scenarios. This will happen more with scale.

At `qa/`:
```shell
$ python generator.py -exp_type programmatic_scenarios -prompt_type programmatic_scenarios -model_type chat -model gpt-4o-mini -max_tokens 1024 -temperature 1 -num_iters 10 -run_name gpt-4o-mini-scenarios
```

This will create a new directory within `CHASE/qa/generation_outputs` with the name `gpt-4o-mini-scenarios` which will store the model generated scenarios in a `scenarios.tsv` file.

### Generate question-answer pairs

We shall use GPT-4o-mini to generate 2 QA pairs for every scenario in the `scenarios.tsv` file.

At `qa/`:
```shell
$ python generator.py -exp_type programmatic_qa -prompt_type programmatic_qa -scenarios_name gpt-4o-mini-scenarios -model_type chat -model gpt-4o-mini -max_tokens 1024 -temperature 1 -num_iters 2 -run_name gpt-4o-mini-qa
```

This will create a new directory within `CHASE/qa/generation_outputs` with the name `gpt-4o-mini-qa` which will store the model generated QA pairs in a `prog_qa.tsv` file.

To ensure that the generations adhere to the compatible format, we must post-process the file.

At `qa/`:
```shell
$  python post_process.py -exp_type programmatic_qa -folder_name gpt-4o-mini-qa -data prog_qa
```

This will create two files in the `CHASE/qa/generation_outputs/gpt-4o-mini-qa` folder: `prog_qa_modified.tsv` and `prog_qa_exceptions.tsv`. The modified file consists of data that was successfully parsed and converted to the compatible format, while the exceptions file consists of the rest of the data. You may use the `-verbose` flag to understand where the exceptions are occuring. Generally, using a stronger instruction-following model such as GPT-4o will yield fewer exceptions.

### Generate adversarial question-answer pairs

We shall use GPT-4o-mini to generate 3 adversarial QA pairs for every qa pair in the `prog_qa_modified.tsv` file.

At `qa/`:
```shell
$ python generator.py -exp_type programmatic_adversarial -prompt_type programmatic_adversarial -questions_name gpt-4o-mini-qa -model_type chat -model gpt-4o-mini -max_tokens 1024 -temperature 1 -num_iters 3 -run_name gpt-4o-mini-adv
```

This will create a new directory within `CHASE/qa/generation_outputs` with the name `gpt-4o-mini-adv` which will store the model generated adversarial QA pairs (as well as the original QA pairs) in a `prog_qa.tsv` file.

To ensure that the generations adhere to the compatible format, we must again post-process the file.

At `qa/`:
```shell
$  python post_process.py -exp_type programmatic_adversarial -folder_name gpt-4o-mini-adv -data prog_qa
```

This will create two files in the `CHASE/qa/generation_outputs/gpt-4o-mini-adv` folder: `prog_qa_modified.tsv` and `prog_qa_exceptions.tsv`. The modified file consists of data that was successfully parsed and converted. We bunch up the original QA pair, and its corresponding adversarial QA pairs in a list as a single row in the modified tsv.

We then perform the LLM-based verification as described in the paper.

At `qa/`:
```shell
$ python verification.py -exp_type programmatic_adversarial -folder_name gpt-4o-mini-adv -data prog_qa_modified -model_type gemini -model gemini-1.5-flash
```

This stores the verified data in the `prog_qa_modified_verified.tsv` file.

### Generate documents

We shall use GPT-4o-mini to generate documents for every qa pair (both original and adversarial) in the `prog_qa_modified_verified.tsv` file.

At `qa/`:
```shell
$ python generator.py -exp_type programmatic_docs -prompt_type programmatic_docs -adversarial_name gpt-4o-mini-adv -model_type chat -model gpt-4o-mini -max_tokens 8000 -temperature 1 -run_name gpt-4o-mini-docs
```

To ensure that the generations adhere to the compatible format, we must again post-process the file.

At `qa/`:
```shell
$  python post_process.py -exp_type programmatic_docs -folder_name gpt-4o-mini-docs -data programmatic_data
```

This will create the `programmatic_data_modified.tsv` file in the `CHASE/qa/generation_outputs/gpt-4o-mini-docs` folder.

We then perform the LLM-based verification as described in the paper.

At `qa/`:
```shell
$ python verification.py -exp_type programmatic_docs -folder_name gpt-4o-mini-docs -data programmatic_data_modified -model_type gemini -model gemini-1.5-flash
```

This stores the verified data in the `programmatic_data_modified_verified.tsv` file.

### Generate final data file

Now that we have generated all the ingredients, we just need to combine the documents to make a unified context for the examples.

At `qa/`:
```shell
$ python cleanup.py -out_dir generation_outputs/gpt-4o-mini-docs/ -og_data programmatic_data_modified_verified -seed_data programmatic_data_modified_verified -threshold 0
```

Here, the `threshold` parameter controls the number of random examples from which you want to add irrelevant documents to a given example. By default it is set to 0. Note that the documents of the adversarial examples are added to the context by default.

The above command will generate the `programmatic_data_modified_verified_cleaned.tsv` file which is the final generated data file.


## IB. Solving CHASE-QA

We will show an example of using the Llama-3.3-70B model to solve the CHASE-QA examples.

Host the Llama model locally using [vllm](https://github.com/vllm-project/vllm) (Assuming 2 A100s):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --max_model_len 16000 --gpu_memory_utilization 0.95
```

Then query the hosted model.

At `qa/`:
```shell
$ python solver.py -data chase_qa -prompt_type zero-shot-basic -model_type vllm -model meta-llama/Llama-3.3-70B-Instruct -run_name chase_qa_llama
```

The output logs and predictions will be stored in the `outputs/chase_qa_llama` folder.

Then for evaluation, use:

At `qa/`:
```shell
$ python evaluator.py -run_name chase_qa_llama -data chase_qa -model_type chat -model gpt-4o
```

## IC. Viewing CHASE-QA

If you want to manually review say the "5th" example from CHASE-QA,

At `qa/`:
```shell
$ python convert_files.py -data_dir data/ -data chase_qa -path readable_data/ -example 5
```

This will create a new directory `readable_data` with a txt file where you will be more easily able to parse the example. You can also set `-example all` to convert the entire dataset into readable txt files.




## IIA. Generating CHASE-Code

Note: Executing the generation or solving files may create some noise files in the directory such as "sandbox*" directories or random data files. These can be deleted after the execution is over.

### Generate helper functions

We shall use GPT-4o-mini to generate helper functions (each iteration attempts to generate 5) for the 'data pre-processing' domain. Note that many iterations will fail because of parsing errors or if the generated helper function fails to execute.

At `code/`:
```shell
$ python generator.py -exp_type helper_functions -prompt_type helper_functions -model_type chat -model gpt-4o-mini -domain data_preprocessing -num_iters 5
```

This will create a new json file within `CHASE/code/helper_functions/generated` with the name `[domain]_new.json` which will store the model generated helper functions. If there already exist helper functions of this domain previously generated/used in a `[domain].json` file, then just manually transfer the newly generated data points from the new file to the old one. If not, just remove `_new` from the name. The next step will expect all helper functions in `[domain].json` file.

### Generate problem statement and answer code

We shall use GPT-4o-mini to generate pairs of (problem statement, answer code) (each iteration attempts to generate 1 pair) for the 'data pre-processing' domain. Our default prompt urges the model to use at least 4 helper functions (out of 10 randomly sampled) in the answer code. Note that many iterations will fail because of parsing errors or if the generated answer code fails to execute.

At `code/`:
```shell
$ python generator.py -exp_type problem_statement -prompt_type problem_statement -model_type chat -model gpt-4o-mini -domain data_preprocessing -num_iters 30 -run_name gpt-4o-mini-problems
```

This will create a new directory within `CHASE/code/generation_outputs` with the name `gpt-4o-mini-problems` which will store the model generated problems in a `problems.tsv` file.

### Generate test code for the problems

We shall use GPT-4o-mini to generate test codes for the problems we generated. The number of iterations reflects the number of attempts (inference calls) to generate a test code for each problem.

At `code/`:
```shell
$ python generator.py -exp_type test -prompt_type test -model_type chat -model gpt-4o-mini -domain data_preprocessing -num_iters 10 -problems_name gpt-4o-mini-problems -run_name gpt-4o-mini-tests
```

This will create a new directory within `CHASE/code/generation_outputs` with the name `gpt-4o-mini-tests` which will store the model generated problems and corresponding tests in a `tested_problems.tsv` file.

### Verify correspondence of problem specification with answer code

We shall use GPT-4o to check if the problem statement correctly specifies the answer code, when provided with the correct set of relevant helper functions.

At `code/`:
```shell
$ python generator.py -exp_type verify_problems -prompt_type verify_problems -model_type chat -model gpt-4o -domain data_preprocessing -num_iters 2 -tested_name gpt-4o-mini-tests -run_name gpt-4o-mini-verified
```

This will create a new directory within `CHASE/code/generation_outputs` with the name `gpt-4o-mini-verified` which will store the verified problems and corresponding tests in a `verified_problems.tsv` file.

### Get final data with large repository contexts

For each problem, we shall create a large repository context across multiple files by sampling 10 random helper functions for each relevant helper function.

At `code/`:
```shell
$ python generator.py -exp_type final_data -domain data_preprocessing -verified_name gpt-4o-mini-verified -run_name gpt-4o-mini-final -extra_fns 10
```

This will create a new directory within `CHASE/code/generation_outputs` with the name `gpt-4o-mini-final` which will store the final data file: `final_data.tsv`.

## IIB. Solving CHASE-Code

We will show an example of using the Llama-3.3-70B model to solve the CHASE-Code examples.

Host the Llama model locally using [vllm](https://github.com/vllm-project/vllm) (Assuming 2 A100s):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --max_model_len 16000 --gpu_memory_utilization 0.95
```

Then query the hosted model.

At `code/`:
```shell
$ python solver.py -data chase_code_dp.tsv -prompt_type basic -model_type vllm -model meta-llama/Llama-3.3-70B-Instruct -run_name chase_code_llama
```

The output logs and predictions will be stored in the `outputs/chase_code_llama` folder.

## IIC. Viewing CHASE-Code

If you want to manually review say the "5th" example from CHASE-Code,

At `code/`:
```shell
$ python convert_files.py -data_dir data/ -data chase_code_dp -path readable_data/ -example 5
```

This will create a new directory `readable_data` with a txt file where you will be more easily able to parse the example. You can also set `-example all` to convert the entire dataset into readable txt files.




## IIIA. Generating CHASE-Math

### Generate problems

If you want to use:
Generator: gpt-4o-mini
Verifier 1: claude-3.5-haiku
Verifier 2: gemini-1.5-flash
Meta-verifier (overseer): Qwen2.5-72B
Grammar corrector: gpt-4o-mini

Host the Qwen model locally using [vllm](https://github.com/vllm-project/vllm) (Assuming 2 A100s):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --max_model_len 4000 --gpu_memory_utilization 0.95
```

At `math/`:
```shell
$ python generator.py -exp_type problem_extend -prompt_type problem_extend -model_type chat -model gpt-4o-mini -ver_model_type_1 anthropic -ver_model_1 claude-3-5-haiku-20241022 -ver_model_type_2 gemini -ver_model_2 gemini-1.5-flash -overseer 1 -overseer_model_type vllm -overseer_model Qwen/Qwen2.5-72B-Instruct -grammar 1 -grammar_model_type chat -grammar_model gpt-4o-mini -num_iters 10 -max_depth 5 -min_depth 2 -max_tokens 512 -seed_name seeds -run_name gpt-4o-mini-problems
```

This will create a directory in `CHASE/math/generation_outputs` with the name `gpt-4o-mini-problems` which will store the model generated problems in `problems.tsv` file.

## IIB. Solving CHASE-Math

We will show an example of using the Llama-3.3-70B model to solve the CHASE-Math examples.

Host the Llama model locally using [vllm](https://github.com/vllm-project/vllm) (Assuming 2 A100s):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.3-70B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --max_model_len 16000 --gpu_memory_utilization 0.95
```

Then query the hosted model.

At `math/`:
```shell
$ python solver.py -data chase_math.tsv -prompt_type 8-shot-cot -model_type vllm -model meta-llama/Llama-3.3-70B-Instruct -run_name chase_math_llama
```

The output logs and predictions will be stored in the `outputs/chase_math_llama` folder.

# Citation

If you use our data or code, please cite our work:

```
@misc{patel2025llmgeneratechallengingproblems,
      title={How to Get Your LLM to Generate Challenging Problems for Evaluation}, 
      author={Arkil Patel and Siva Reddy and Dzmitry Bahdanau},
      year={2025},
      eprint={2502.14678},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14678}, 
}
```

For any clarification, comments, or suggestions please contact [Arkil](http://arkilpatel.github.io/).
