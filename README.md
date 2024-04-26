# Ray Handler Package

A package for distributed computation using ray.

Handler reads arguments from the command line for each group:
 - Parameters that determine the output of primary function evaluations of the stages of a script.
 - Options that do not affect the output of primary function evaluations of the stages of a script, but affect how they are calculated or plotted.
 - Handler-specific options that determine the behavior of the handler when running the stages of a script.

The computation is split into stages, whose progress is periodically saved allowing the computation to be interrupted and resumed later.

Three kinds of stages are included:
 - SingleStage: Runs a function and saves the results.
 - MultiStage: Runs a function over multiple inputs, in parallel if desired, periodically saving the results so the computation can be interrupted and resumed.
 - PlotStage: Runs a function every time, intended for plotting results.