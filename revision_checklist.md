# Numerical-experiments revision checklist

Goal: make every subsection of Section 6 follow the same narrative arc as
6.1.1 (Isolated / Newtonian), which is the reference "done nicely" case.

## The six-beat template

1. **Setup + purpose** — geometry, forcing, fluid, true parameters, and one line
   on what the case tests.
2. **FOM data** — simulate to `t_end`; state the number of sampling points `N`.
3. **Noise ladder** — zero-mean Gaussian at a stated fraction of `Δx_max`.
4. **SAM-vs-FOM discrepancy decision** — compare SAM at ground truth to FOM, then
   explicitly say whether the model-bias term is kept or dropped.
5. **Priors** — table + short justification.
6. **Posterior** — results table + one-line takeaway.

Beat 4 is the connective tissue; it is the beat most often missing or implicit
in the later subsections.

## Per-subsection status

### 6.1.1 Newtonian (reference) — complete
All six beats present and in order. Use as the model.

### 6.1.2 Isolated viscoelastic
- [ ] Fix beat-4 copy-paste bug: text says "infer only η_s and σ_exp" — should be
      **η_s, η_p, λ and σ_exp**.
- [ ] Fix cross-reference "As Fig. 6 shows" → **Fig. 7**.
- [ ] (optional) mirror the corner-plot correlation note in other multi-parameter cases.

### 6.1.3 Nonlinear viscoelastic
- [ ] Beats 2–3: state `N = 51` and the 2% noise sentence explicitly up front.
- [ ] Beat 4 flips here: SAM at ground truth does NOT match FOM — say so explicitly
      and state that the bias term is retained (mirror of the Newtonian decision).
- [ ] Beats 5–6 (Tables 7, 8, 9) OK.

### 6.2.1 Wall / Newtonian
- [ ] Add beat-4 line up front: SAM at ground truth matches FOM (Fig. 11), no bias term.
- [ ] State the 2% noise + `N = 30` in the reference voice.
- [ ] Label the κ/s identifiability and unbounded-model-error studies as *beyond* the
      baseline template so they don't read as part of the core recipe.

### 6.2.2 Wall / viscoelastic
- [ ] Restate `N = 51` and the 2% noise sentence in this subsection.
- [ ] Add beat-4 line: Fig. 15 shows FOM/SAM agreement, so no bias term.
- [ ] Beats 5–6 (Table 12, Fig. 16, Table 13) OK.

### 6.3.1 Homogeneous material
- [ ] Frame Table 15 (uncorrected vs Hasimoto) explicitly as the beat-4 SAM/model-adequacy check.
- [ ] (optional) add a posterior-spread element for G.

### 6.3.2 Heterogeneous material
- [ ] Add explicit priors (Table 14) and noise-level callouts even though the 6.3 intro covers them.
- [ ] Beat 6 (Fig. 18, Eqs. 39–40) OK; consider a small summary table analogous to Table 13.

## The To-Do note itself
The note "FOM data → added noise → SAM performance → priors → posterior" is beats
2→3→4→5→6. It omits beat 1 (setup/purpose), which 6.1.1 opens with. Expand the note
to the six-step list above, then delete it once every subsection passes.
