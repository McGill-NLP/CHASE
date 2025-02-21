## A. Generating CHASE-Math

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

## B. Solving CHASE-Math

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
