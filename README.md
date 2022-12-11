# Uncertainty-Aware-DRL

The only files you need to run are 'main_u.py' and 'main_ql.py'
'main_ql.py' is to evaluate the effectiveness and efficiency of our algorithm, which will give you the accuracy of our algorithm, and figures about how our model is converged.
'main_u.py' is to quantify the uncertainties such as vacuity, dissonance, and entropy

How to run:
```
For 'main_ql.py', simply run 'python3 main_ql.py'
It will give you accuracy for each setting we have for algorithm as well as their figures
```

```
For 'main_u.py', run 'python3 main_ql.py'
It will show the figure of Shannon entropy with our default settings.
If you want to see other figures, simply uncomment the corresponding lines of code.
For example, uncomment uncertain() will give you the figure of vacuity for each setting of our algorithm
draw_d() is for dissonance
draw_e() is for entropy
```
