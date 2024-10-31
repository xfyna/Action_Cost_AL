# Action_Cost_AL
**Cost-Aware Query Policies in Active Learning for Efficient Autonomous Robotic Exploration**

*Abstract*

In missions constrained by finite resources efficient data collection is critical. Informative path planning, driven by automated decision-making, optimizes exploration by reducing the costs associated with accurate characterization of a target in an environment. Previous implementations of active learning (AL) did not consider the action cost for regression problems or only considered the action cost for classification problems. This paper analyzes an AL algorithm for Gaussian Process (GP) regression while incorporating action cost. The algorithm’s performance is compared on various regression problems to include terrain mapping on diverse simulated surfaces along metrics of root mean square (RMS) error, samples and distance until convergence, and model variance upon convergence. The cost-dependent acquisition policy doesn’t organically optimize information gain over distance; instead, the traditional uncertainty metric with a distance constraint best minimizes root-mean-square error over trajectory distance. This study’s impact is to provide insight into incorporating action cost with AL methods to optimize exploration under realistic mission constraints.

Distance-Constrained AL Trials
--> distance-constrained_AL.py

Distance-Normalized AL Trials (With 1dx/3dx Comparison)
--> distance-normalized_AL.py

Conventional AL Trials
--> conventional_unconstrained_AL.py

Surface Complexity Calculation
--> surface_complexity.py

**Important Note:
You must have the Lunar Crater .TIF files installed and located in the same directory as the above AL trial code when running a trial (find all Lunar Crater .TIF files here: https://github.com/xfyna/AL-BNN-GP.git)

In order to run any trials, ensure the following system arguments are entered:

[1] Surface Type ("Parabola", "Lunar", "Townsend")

[2] Trial Name (i.e., a number, letter, etc. that will be appended to the newly created trial folder)
