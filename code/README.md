## A. Generating CHASE-Code

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

## B. Solving CHASE-Code

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

## C. Viewing CHASE-Code

If you want to manually review say the "5th" example from CHASE-Code,

At `code/`:
```shell
$ python convert_files.py -data_dir data/ -data chase_code_dp -path readable_data/ -example 5
```

This will create a new directory `readable_data` with a txt file where you will be more easily able to parse the example. You can also set `-example all` to convert the entire dataset into readable txt files.
