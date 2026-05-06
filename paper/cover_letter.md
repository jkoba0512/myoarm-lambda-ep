<!--
File          : paper/cover_letter.md
Use           : Cover letter accompanying submission of the manuscript
                "Decoupling smoothness, accuracy, and kinematic invariance
                in biological reach: an ablation study of an
                equilibrium-point controller in a 34-muscle arm model"
                to Biological Cybernetics (Springer).
Status        : DRAFT 2026-05-06
-->

May 6, 2026

Prof. Benjamin Lindner, Prof. Peter Thomas, Prof. Jean-Marc Fellous,
and Prof. Paul Tiesinga
Editors-in-Chief
*Biological Cybernetics*
Springer Nature

---

Dear Editors,

I am writing to submit the enclosed manuscript, **"Decoupling smoothness, accuracy, and kinematic invariance in biological reach: an ablation study of an equilibrium-point controller in a 34-muscle arm model,"** for consideration as an **Original Article** in *Biological Cybernetics*.

The paper asks a classical motor-control question — *which biological mechanisms are responsible for which kinematic invariants of human reach?* — in a high-dimensional musculoskeletal setting that has only recently become tractable. Using the MyoSuite myoArm (20 degrees of freedom, 34 Hill-type muscles), I implement a layered biologically motivated controller combining (i) Feldman's λ-equilibrium-point hypothesis with (ii) a minimum-jerk virtual trajectory $\boldsymbol{\lambda}(t)$, (iii) a 200 ms visuomotor correction, and (iv) γ-compatible spinal reflexes (Ia, Ib, reciprocal inhibition). I then run a factorial ablation that *attributes separable kinematic axes to distinct biological control layers*: the virtual trajectory primarily shapes smoothness, visuomotor feedback primarily shapes accuracy, and stretch reflexes primarily shape velocity-peak timing. The full controller is practically equivalent in endpoint accuracy to an endpoint-PD baseline (Cohen's $d = +0.03$ on $\approx 100$ mm minimum tip error, within a pre-defined $\pm 20$ mm equivalence margin) while halving peak speed and reducing jerk by 40 %. Only the variant with stretch reflexes brings the velocity-peak time ratio into the canonical human range (0.40–0.50). I also report a negative result: an online cerebellar correction in either joint or λ space did not improve performance, which is consistent with viewing the cerebellum as a slow inverse-model learner rather than a within-trial steering controller.

I believe this work fits *Biological Cybernetics* squarely on three counts. First, the journal has historically been the natural home for theoretical and biomimetic studies of the equilibrium-point hypothesis and of trajectory-formation principles (Feldman, Bizzi, Hogan, Flash & Hogan, Latash, Kawato, and others). Second, the paper offers a *mathematical description of the processes underlying neuronal functioning* in the journal's stated sense, by providing a quantitative ablation-based decomposition of which kinematic invariant arises from which control layer. Third, it offers a *biomimetic implementation of neuronal processing strategies* in a 34-muscle arm — a regime in which most prior λ-EP studies have been theoretical or low-DoF. The negative result on the cerebellar correction is, I think, a useful counterpoint that the journal's readership is well placed to interpret.

I want to draw the editors' attention to two further points. (a) During this work I uncovered a small but consequential reproducibility bug in the MyoSuite reach environments (in the versions tested), in which `env.reset(seed=N)` does not by itself restore deterministic targets. I release a `deterministic_reset` helper that fixes this; I expect this to be of independent value to the readership. (b) Because the comparison condition matters in this kind of paper, I deliberately keep the engineering baseline simple (endpoint-PD + spinal) so that the non-trivial differences (peak speed, jerk, velocity-peak ratio) cannot be attributed to PD-vs-controller asymmetries rather than to the biological layers under study.

I confirm that this manuscript is original work, has not been published elsewhere, and is not under consideration at any other journal. There are no competing interests. I am the sole author. A preprint of this manuscript is posted on bioRxiv ([10.64898/2026.05.01.722167](https://doi.org/10.64898/2026.05.01.722167), 2026-05-06); I understand from the Springer Nature preprint policy that this does not constitute prior publication. The complete implementation, exact seed lists, raw per-seed metric tables, trained CfC weights, and figure-generation pipeline are publicly archived under the MIT license at GitHub `jkoba0512/myoarm-lambda-ep` and Zenodo [10.5281/zenodo.19948021](https://doi.org/10.5281/zenodo.19948021) (release `v1.0.0-bioRxiv`).

Thank you for considering this manuscript. I look forward to the editorial assessment.

Sincerely,

**Jun Kobayashi**
Department of Intelligent and Control Systems
Graduate School of Computer Science and Systems Engineering
Kyushu Institute of Technology
680-4 Kawazu, Iizuka, Fukuoka 820-8502, Japan
ORCID: [0009-0002-6318-2617](https://orcid.org/0009-0002-6318-2617)
E-mail: kobayashi.jun184@m.kyutech.ac.jp
