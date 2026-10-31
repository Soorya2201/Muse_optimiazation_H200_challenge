# Scheduling research addition

[Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs](https://arxiv.org/abs/2512.18134v1),
Rupanshu Soi et al., Dec19,2025; [full text](https://arxiv.org/html/2512.18134v1),
CC BY4.0. Abstract and relevant scheduling/code-generation sections reviewed.

Twill jointly models software pipelining and warp assignment using solver
constraints, including memory/resource limits. Its optimality is relative
to the encoded machine model and supported loop class; it does not prove
an arbitrary whole Muse layer globally optimal. The evaluation recovers
expert attention schedules on Hopper and Blackwell.

Application here is a research proposal, not an implemented optimization:
model the measured register lifetimes, asynchronous dependencies and unit
publication before changing warpgroup count again. Our PIPE3 experiment
shows that overlap helping a weaker two-group control can still lose to
the three-group incumbent. Missing measured operation costs and irregular
expert-control flow remain unresolved; do not substitute guessed constants
for H200 evidence or claim a solver-derived performance gain.

Status: no Twill implementation downloaded/executed, no GPU run allocated
solely on the paper, no canonical or scoring change.
