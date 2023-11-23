# Building the Maps

To reproduce the results from our paper, simply run reproduce_results.py.
To experiment with other language to KG mappings, define a new configuration file in the
configs directory and pass it to the training script through run_linear_model.py {path-to-config}.

```math
\begin{tabular}{lrrr}
\toprule
                                             names &        h5 &       h10 &   avg\_sim \\
\midrule
 infer1 &  0.055836 &  0.080933 &  0.388901 \\
 laser &  0.095375 &  0.128651 &  0.249553 \\
 skipthought &  0.017670 &  0.025674 &  0.269965 \\
 gem &  0.241005 &  0.311117 &  0.508135 \\
 random &  0.076323 &  0.094632 &  0.119249 \\
 quickthought &  0.015632 &  0.022333 &  0.484893 \\
 infer2 &  0.059475 &  0.084826 &  0.317696 \\
 glove &  0.119046 &  0.151657 &  0.526276 \\
 dct_1 &  0.000061 &  0.000079 &  0.190908 \\
 dct_5 &  0.024301 &  0.033958 &  0.147522 \\
 sentbert &  0.104674 &  0.131485 &  0.481374 \\
\bottomrule
\end{tabular}
```
