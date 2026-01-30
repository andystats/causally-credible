# Credence Framework Tutorial App

**Module 5: Companion Tool Guide**

🔗 **Live App:** [https://andystats.shinyapps.io/causally-credible_tutorial/](https://andystats.shinyapps.io/causally-credible_tutorial/)

---

## Overview

The **Credence framework** (Parikh et al., ICML 2022) generates semi-synthetic data with known ground truth causal effects. Unlike traditional simulations that use parametric assumptions, Credence learns realistic covariate structure from actual data using a Conditional VAE.

### Key Innovation: Separate REALISM from TRUTH

| Concept | Description |
|---------|-------------|
| **Realism** | Learn p(X\|Z) from empirical data |
| **Truth** | Inject known ATE (τ) and unmeasured confounding (ρ) |
| **Result** | Credible benchmarks to validate causal methods |

---

## Why Credence?

Traditional simulations (like simcausal in Module 4) have limitations:

- Hand-specified DGPs embed assumptions that may favor certain estimators
- - Parametric models rarely match real-world complexity
  - - Hard to know if method comparisons generalize to actual data
   
    - **Credence addresses this by:**
   
    - - Anchoring covariate distributions to empirical data
      - - Providing tunable "bias knobs" for controlled experiments
        - - Enabling fair comparisons under realistic conditions
         
          - ---

          ## Getting Started

          Navigate to the app—no installation required:

          👉 [https://andystats.shinyapps.io/causally-credible_tutorial/](https://andystats.shinyapps.io/causally-credible_tutorial/)

          ---

          ## App Workflow: 4 Steps

          ### Step 1: Load Observational Data

          Upload your own CSV or use built-in examples.

          **Example Datasets:**
          - **LaLonde NSW:** Classic job training program evaluation
          - - **Pneumonia Vaccine (M4):** The dataset from Module 4
           
            - **What happens:**
            - - Select your outcome variable (Y)
              - - Select your binary treatment variable (Z)
                - - All other variables become covariates (X)
                  - - The app learns p(X|Z)—how covariates differ between treated/control
                   
                    - **Requirements:**
                    - - Z must be binary (0/1)
                      - - Y should be numeric
                        - - Missing values are dropped (complete-case analysis)
                         
                          - ---

                          ### Step 2: Learn Data Structure

                          Train a Conditional VAE to model p(X|Z).

                          **Parameters:**
                          - **Training Epochs:** More epochs = better fit (default: 50)
                         
                          - **Under the hood:**
                          - - **Encoder:** Compresses (X, Z) into latent space U
                            - - **Decoder:** Reconstructs X from (U, Z)
                              - - **ELBO loss:** Balances reconstruction fidelity with regularization
                               
                                - **Why VAE?**
                                - - Learns complex, non-linear covariate relationships
                                  - - Generates new samples that look like real data
                                    - - Captures how X differs between treatment groups
                                     
                                      - ---

                                      ### Step 3: Generate Biased Synthetic Data

                                      Create new data with known ground truth.

                                      **Key Parameters:**

                                      | Parameter | Description | Range |
                                      |-----------|-------------|-------|
                                      | **τ (tau)** | True ATE—the causal effect you want to inject | -10 to +10 |
                                      | **ρ (rho)** | Unmeasured confounding strength via latent U | 0.0 to 1.0 |
                                      | **Sample Size** | Number of synthetic observations | Default: 1500 |

                                      **Understanding ρ (rho):**
                                      - `ρ = 0`: No unmeasured confounding (exchangeability holds)
                                      - - `ρ > 0`: U affects both treatment assignment and outcome
                                       
                                        - **Data Generating Process:**
                                       
                                        - ```
                                          Treatment:  P(Z=1|U) = logistic(α + ρ·U)
                                          Outcomes:   Y(0) = f(X) + γ·U + ε
                                                      Y(1) = Y(0) + τ
                                          True ATE = τ (by construction)
                                          ```

                                          ---

                                          ### Step 4: Analyze & Export

                                          Compare naive estimates to ground truth.

                                          **What you'll see:**
                                          - **True ATE (τ):** The value you set
                                          - - **Naive Estimate:** Simple difference in means
                                            - - **Bias:** How far naive is from truth
                                             
                                              - **Why naive is biased:**
                                             
                                              - ```
                                                Naive = τ + [E(Y(0)|Z=1) - E(Y(0)|Z=0)]
                                                ```

                                                If confounders shift Y(0) across groups, bias appears.

                                                **Download:** Export synthetic dataset as CSV and use with Module 4 methods to test estimator performance.

                                                ---

                                                ## Practice Exercises

                                                ### Exercise 1: Validate TMLE Under No Confounding

                                                1. Load the Pneumonia Vaccine example
                                                2. 2. Train the CVAE (50 epochs)
                                                   3. 3. Set τ = -5, ρ = 0 (no unmeasured confounding)
                                                      4. 4. Generate 2000 samples
                                                         5. 5. Download data and analyze with Module 4 app
                                                            6. 6. **Verify:** TMLE should recover τ = -5
                                                              
                                                               7. ### Exercise 2: Stress Test Under Hidden Bias
                                                              
                                                               8. 1. Same setup, but set ρ = 0.3
                                                                  2. 2. Generate data and analyze
                                                                     3. 3. **Observe:** How does TMLE perform now?
                                                                        4. 4. Try increasing ρ to 0.5, then 0.7
                                                                           5. 5. **Question:** At what point does TMLE break down?
                                                                             
                                                                              6. ### Exercise 3: Compare Estimator Robustness
                                                                             
                                                                              7. 1. Generate data with moderate confounding (ρ = 0.4)
                                                                                 2. 2. Apply multiple methods: G-comp, IPW, TMLE, DML
                                                                                    3. 3. **Compare:** Which methods are most robust?
                                                                                       4. 4. Repeat with different τ values
                                                                                         
                                                                                          5. ### Exercise 4: Use Your Own Data
                                                                                         
                                                                                          6. 1. Upload a CSV from your research domain
                                                                                             2. 2. Train CVAE to learn your covariate structure
                                                                                                3. 3. Inject known effects and test your analysis pipeline
                                                                                                   4. 4. **Validate:** Can your methods recover ground truth?
                                                                                                     
                                                                                                      5. ---
                                                                                                     
                                                                                                      6. ## Key Concepts
                                                                                                     
                                                                                                      7. ### Observed vs Unobserved Confounding
                                                                                                     
                                                                                                      8. | Type | Description |
                                                                                                      9. |------|-------------|
                                                                                                      10. | **Observed** | Captured in p(X\|Z) learned from data |
                                                                                                      11. | **Unobserved** | Injected via U with strength ρ |
                                                                                                     
                                                                                                      12. Adjusting for X handles observed confounding; ρ tests robustness to unmeasured confounding.
                                                                                                     
                                                                                                      13. ### The CVAE Components
                                                                                                     
                                                                                                      14. - **Encoder q(U|X,Z):** Compresses data to latent space
                                                                                                          - - **Decoder p(X|U,Z):** Reconstructs covariates
                                                                                                            - - **ELBO:** Evidence Lower Bound optimized during training
                                                                                                             
                                                                                                              - ### Bias Knobs
                                                                                                             
                                                                                                              - - **ρ (rho):** Unmeasured confounding strength
                                                                                                                - - **τ (tau):** True causal effect (the signal)
                                                                                                                  - - *(Advanced: positivity trimming, measurement error)*
                                                                                                                   
                                                                                                                    - ---
                                                                                                                    
                                                                                                                    ## Connecting to Other Modules
                                                                                                                    
                                                                                                                    ### From Module 2 (Causal Roadmap)
                                                                                                                    
                                                                                                                    - **Step 5:** Assess identifiability assumptions
                                                                                                                    - - Credence lets you test what happens when assumptions are violated
                                                                                                                     
                                                                                                                      - ### From Module 4 (Estimation)
                                                                                                                     
                                                                                                                      - - Apply G-computation, IPW, TMLE to Credence data
                                                                                                                        - - Compare estimator performance against known truth
                                                                                                                          - - Test doubly-robust properties under model misspecification
                                                                                                                           
                                                                                                                            - ### The Pipeline
                                                                                                                           
                                                                                                                            - ```
                                                                                                                              1. Use Credence to generate benchmark data
                                                                                                                              2. Analyze with Module 4 methods
                                                                                                                              3. Compare estimates to known τ
                                                                                                                              4. Stress test by increasing ρ
                                                                                                                              ```
                                                                                                                              
                                                                                                                              ---
                                                                                                                              
                                                                                                                              ## Interpretation Guidance
                                                                                                                              
                                                                                                                              ### When ρ = 0
                                                                                                                              
                                                                                                                              - Exchangeability holds (given X)
                                                                                                                              - - Consistent methods should recover τ
                                                                                                                                - - Differences reflect model specification, not confounding
                                                                                                                                 
                                                                                                                                  - ### When ρ > 0
                                                                                                                                 
                                                                                                                                  - - Unmeasured confounding exists
                                                                                                                                    - - All methods will be biased to some degree
                                                                                                                                      - - Tests robustness, not correctness
                                                                                                                                        - - Useful for sensitivity analysis
                                                                                                                                         
                                                                                                                                          - ### What to Report
                                                                                                                                         
                                                                                                                                          - - The τ and ρ settings used
                                                                                                                                            - - How estimates compare to truth
                                                                                                                                              - - At what ρ level methods diverge significantly
                                                                                                                                               
                                                                                                                                                - ---
                                                                                                                                                
                                                                                                                                                ## References
                                                                                                                                                
                                                                                                                                                Parikh H, Vajao C, Xu L, Tchetgen Tchetgen E (2022). *Validating Causal Inference Methods.* Proceedings of the 39th International Conference on Machine Learning (ICML). [arXiv:2202.04208](https://arxiv.org/abs/2202.04208)
                                                                                                                                                
                                                                                                                                                Kingma DP, Welling M (2014). *Auto-Encoding Variational Bayes.* ICLR.
                                                                                                                                                
                                                                                                                                                Higgins I, et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
