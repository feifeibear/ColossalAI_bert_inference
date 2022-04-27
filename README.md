## An example to show how ColossalAI helps you write distributed inference code.

the file contains a script to run 1D Tensor Parallelism (also known as Model Parallelism) on 4 GPUs.


## The known issues
If you met the following errors.
```
IndexError: tuple index out of range
```

Replacing the following line of code in the modeling_bert.py of transformers library.
```
1008 # sequence_output = encoder_outputs[0]
1009 sequence_output = encoder_outputs.last_hidden_state

```
