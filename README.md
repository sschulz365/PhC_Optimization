# PhC_Optimization
This is a multi-objective optimization algorithm for the design of PhC waveguides. The optimization algorithm is implemented in python, but the simulations of the PhC waveguides are performed using the MIT photonics band package (MPB, http://ab-initio.mit.edu/wiki/index.php/MIT_Photonic_Bands), and use the effective period method (http://iopscience.iop.org/article/10.1088/2040-8978/17/7/075006/meta and download the necessary scripts here https://github.com/sschulz365/PhC_simulations/tree/2D-simulations).

The algorithm combines a strenght pareto evolutionary algorithm (SPEA) with a relatively steep gradient descent (RS) and is referred to as SPEARS. For more info please read our accompanying publication (http://iopscience.iop.org/article/10.1088/2040-8978/18/11/115005/meta)

All these software packages (SPEARS, MPB and the effective period scripts) are released under the GNU Public license (http://www.gnu.org/copyleft/gpl.html) version 2 and can be redistributed under the same license or a newer version of the GNU Public license. For redistribution of the simulations scripts and SPEARS under a different license, please contact Dr. Schulz (sschulz_at_uottawa.ca or sebastianandreasschulz_at_gmail.com, replace _at_ with @).

When publishing results obtained through this simulation software, please include the following references and a statements to a similar effect acknowleding the authors of all software packages:

For MPB:

Steven G. Johnson and J. D. Joannopoulos, "Block-iterative frequency-domain methods for Maxwell's equations in a planewave basis," Optics Express 8, no. 3, 173-190 (2001), http://www.opticsexpress.org/abstract.cfm?URI=OPEX-8-3-173 

If you want a one-sentence description of the algorithm for inclusion in a publication, we recommend:
Fully-vectorial eigenmodes of Maxwell's equations with periodic boundary conditions were computed by preconditioned conjugate-gradient minimization of the block Rayleigh quotient in a planewave basis, using a freely available software package [ref]. 

For the effective period approximation:
S. A. Schulz,A. H. K. Park, J. Upham and R. W. Boyd "Beyond the effective index method: improved accuracy for 2D simulations of photonic crystal waveguides", Journal of Optics 17, 075006 (2015).
Suggested one sentence description: 
During 2-dimensional simulations, the 3-dimensional device structure was approximated using the effective period approximation [ref].

For SPEARS:
S. Billings, S. A. Schulz, J. Upham, and R. W. Boyd "Application-tailored optimisation of photonic crystal waveguides", Journal of Optics 18, 115005 (2016).

Suggested one sentence description: 
Optimization of the device design was performed using the SPEARS multi-objective optimization algorithm [ref].
