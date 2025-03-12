# HyperParamBRDF: A Parameter-Conditioned Extension of HyperBRDF

**HyperParamBRDF** is a novel extension of the original **HyperBRDF** framework, enabling parameter-conditioned reflectance synthesis for real or synthetic materials. Specifically, it introduces explicit \emph{global parameters} (e.g., thickness or doping levels) to drive the generation of unique BRDFs, whereas the original HyperBRDF treated each material mostly as a stand-alone instance.

---

## 1. Background

**HyperBRDF** was designed to learn a hypernetwork that could generate (or adapt) a local reflectance model for each material in a dataset. This approach allowed for flexible BRDF representations, but it did not inherently account for user-defined parameters such as nano-material thickness or doping. 

In many real-world scenarios—particularly in nano-fabrication—designers want to **interpolate** or **extrapolate** reflectances for parameter combinations not explicitly measured in the training set. HyperParamBRDF addresses this need by introducing a parameter vector (e.g., `[thickness, doping]`) that the hypernetwork recognizes at inference time.

---

## 2. What’s New in HyperParamBRDF

1. **Parameter Inputs**  
   - Each material is now associated with a vector of real-world properties (thickness, doping, or any other relevant attributes).  
   - The hypernetwork conditions on these properties to generate partial parameters or complete BRDF outputs.

2. **Synthetic Data Integration**  
   - To cover a broader space of parameter values, we incorporate synthetic data (e.g., physically-based simulations) alongside real measurements.  
   - This allows the model to generalize better to new parameter combos.

3. **Workflow**  
   - **Training** remains similar to the original HyperBRDF approach, but each material in the dataset is labeled with its parameter vector.  
   - **Inference** (test) now allows specifying *new* parameter vectors directly. The model produces a predicted BRDF for those parameters, even if they were never explicitly measured.

4. **Compatibility**  
   - The partial inference / full reconstruction pipeline from HyperBRDF is retained.  
   - Checkpoints, logs, and the MERL format remain usable if you wish to keep the same artifact naming (e.g., `.pt` → `.fullbin`).

---

## 3. Main Components

1. **Parameter-Conditioned Hypernetwork**  
   - Extended from the standard HyperBRDF “hypernetwork + local net” approach.  
   - Accepts both the usual angle coordinates (for half/diff inference) \emph{and} a parameter vector describing the material.

2. **Dataset Changes**  
   - Each material or synthetic sample now has metadata: `[params, BRDF]`.  
   - We store these in a structure like `MerlDataset` or a new `NanoDataset` that references parameter values from the filename or a sidecar metadata file.

3. **Scripting**  
   - `train.py` accepts `--params` or a `NanoDataset` branch to load parameter vectors.  
   - `test.py` supports specifying new parameter combos. If using the standard code path, you might create dummy `.binary` files with the desired parameter combos, or write a custom script that directly calls the hypernetwork with `[thickness, doping]`.

---

## 4. Installation & Usage

1. **Clone** or download this extended repository:
   ```bash
   git clone https://github.com/username/HyperParamBRDF.git
   cd HyperParamBRDF

	2.	Install Requirements
	•	Python 3.8+
	•	PyTorch, NumPy, SciPy, etc. (see requirements.txt)

pip install -r requirements.txt


	3.	Train
	•	Provide a dataset of materials, each with a parameter vector.
	•	For example:

python main.py \
  --destdir results/nano_experiment \
  --dataset NANO \
  --params data/params.csv \
  --epochs 100


	•	Adjust arguments (like --kl_weight, --fw_weight) as needed.

	4.	Test / Inference
	•	Either use the original test.py + pt_to_fullmerl.py flow with newly created “dummy” .binary files reflecting new parameter combos.
	•	Or write a small custom inference script that sets [thickness, doping] in code:

# pseudo-code
params = torch.tensor([thickness, doping])
brdf = model.forward_angles_and_params(angles, params)


	•	Convert partial results into .fullbin for final MERL-compatible output.

⸻

5. Known Limitations
	•	Parameter Overlap: If you trained primarily on one doping range, extrapolation to significantly larger doping might be unreliable.
	•	Dummy File Trick: If using the old script flow, you still rely on placeholders to specify new parameter combos. A more direct approach is recommended for a truly parameter-driven inference.

⸻

6. Related Work
	•	HyperBRDF: The original approach for learning a hypernetwork across multiple measured BRDFs.
	•	MERL Format: Standard binary format from Matusik et al., using Rusinkiewicz half/diff angle parameterization.
	•	Synthetic Data: For some nano-materials, we include reflectances from physically-based simulation to fill gaps in the measured range.

⸻

7. Contributing

We welcome pull requests or discussions on:
	•	More robust parameter-based inference scripts.
	•	Improved data ingestion (e.g., reading param combos from .csv or .json).
	•	Enhancements to synthetic data generation or new domain-specific parameters.

⸻

8. License & Citation

HyperParamBRDF is released under the same license as HyperBRDF (MIT, BSD, or whichever applies). If you use this extension in published research, please cite both the original HyperBRDF paper/repository and any references that introduced the parameter-driven approach for advanced materials.

@inproceedings{hyperparambrdf2023,
  author    = {Doe, Jane and ...},
  title     = {HyperParamBRDF: A Parameter-Driven Extension of HyperBRDF},
  booktitle = {Under Review...},
  year      = {2023}
}



⸻

Thank you for exploring HyperParamBRDF! We hope it simplifies your workflow for nano-material reflectance prediction, bridging the gap between measured data and advanced parameter-driven reflectance modeling.
