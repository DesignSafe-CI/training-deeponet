# Deep Operator Network (DeepONet)

> Somdatta Goswami

Traditional numerical modeling techniques, while accurate, often prove limiting for real-time predictions due to their computational intensity. Machine learning (ML)-based surrogate models have emerged as a faster alternative. However, these surrogates are constrained by their training data. For instance, an ML model trained to predict structural displacements based on specific earthquake records is limited to that finite dataset, as it fails to capture the underlying relationship between earthquake loading (input functions) and displacement (output functions). Neural Operators address this limitation by learning function-to-function mappings. A notable example in this category is the Deep Operator Network (DeepONet). The DeepONet architecture comprises two neural networks: a branch net that processes the input function, and a trunk net that handles the locations where the output function is evaluated. This structure enables DeepONet to learn the mapping between entire function spaces, rather than just specific input-output pairs. In the context of structural analysis, DeepONet can potentially learn the general relationship between loading conditions and the resulting deflected shapes of beams. This approach promises greater flexibility and generalization capability compared to traditional analytical methods, that needs to be simulated for every loading condition. This Git repository contains codes for the DeepONet to map a random loading condition (displacement-controlled) to the deflection of a cantilever beam. 
<p align="center">
  <img src="https://github.com/DesignSafe-Training/deeponet/blob/main/Schematic.png" alt="Schematic" width="900"/>
  <br/>
  <strong>Schematic</strong>
</p>

For instructions on how to start a GPU Jupyter, refer to [DesignSafe user guide](https://www.designsafe-ci.org/user-guide/tools/jupyterhub/#launch-the-jupyter-lab-hpc-gpu)

