## 1A. Generating CHASE-QA

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


## 1B. Solving CHASE-QA

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

## 1C. Viewing CHASE-QA

If you want to manually review say the "5th" example from CHASE-QA,

At `qa/`:
```shell
$ python convert_files.py -data_dir data/ -data chase_qa -path readable_data/ -example 5
```

This will create a new directory `readable_data` with a txt file where you will be more easily able to parse the example. You can also set `-example all` to convert the entire dataset into readable txt files.
