# ApJL Letter Draft: Resolution of the CatWISE/Secrest Quasar Dipole as a Selection Effect

This file contains a **ready-to-paste ApJL (AAS) letter draft** in AASTeX, based on the analysis
artifacts in `Q_D_RES/`.

You said you will place figures in the **Overleaf project root**. The figures referenced by the
draft below are already present in this folder and can be copied directly:

- `fixed_axis_scaling_fit.png` (primary figure; 3-panel summary)
- `dipole_vs_w1covmin.png` (optional robustness figure)
- `glm_cv_axes_nexp_offset.png` (optional robustness figure)
- `axis_alignment_mollweide.png` (optional context figure)

If you want the **shortest, cleanest ApJL letter**, keep only Figure 1 (`fixed_axis_scaling_fit.png`)
in the main text and move the rest to an online supplement.

---

## AASTeX 6.31 ApJL Draft (paste into Overleaf as `main.tex`)

```tex
%% ApJL Letter draft (AASTeX 6.31)
%% Place figures (PNG) in the Overleaf project root.

\documentclass[twocolumn]{aastex631}

\usepackage{amsmath}

\shorttitle{Selection Origin of the CatWISE Quasar Dipole}
\shortauthors{Smith}

\begin{document}

\title{Evidence that the CatWISE Quasar Dipole Is Dominated by a Faint-Limit Selection Gradient}

\author{Aiden Smith}
\affiliation{Independent Researcher}
\email{aidenblakesmithtravel@gmail.com}

%% ApJL abstracts should be concise; aim <250 words.
\begin{abstract}
Secrest et al.\ (2022, ApJL, 937, L31) reported a dipole in the angular number counts of CatWISE
AGN candidates whose amplitude exceeds expectations from the kinematic dipole.
Using the publicly released accepted CatWISE sample (CatWISE2020; \citealt{catwise2020}), I reproduce a dipole with a consistent
direction and show that its measured amplitude is strongly sensitive to the survey selection
boundary at the faint $W1$ magnitude limit.
Across 21 faint-limit choices $W1_{\max}\in[15.6,16.6]$, the dipole amplitude and its projection
onto the reported dipole axis vary in a manner tightly correlated with the local faint-end
count slope $\alpha_{\rm edge}\equiv \mathrm{d}\ln N/\mathrm{d}m$ evaluated at $W1_{\max}$.
A two-component scaling model in which the observed dipole vector is the sum of a non-scaling term
and a selection term proportional to $\alpha_{\rm edge}$ fits the data with
$\chi^2/\mathrm{dof}\simeq1.10$ and implies an effective dipolar modulation of the faint limit
of $\delta m_{\rm amp}=(-0.0125\pm0.0027)\,\mathrm{mag}$ along the dipole axis.
Furthermore, the inferred ``cleaned'' dipole direction is not stable under plausible changes in
scan-depth modeling: replacing a catalog-derived coverage proxy with an independent unWISE depth
template \citep{unwise_lang2014,unwise_meisner2017} changes the cleaned direction substantially.
These results strongly favor an instrumental/selection origin for the CatWISE dipole and caution
against interpreting its raw amplitude as evidence for a breakdown of statistical isotropy without
an end-to-end completeness model.
\end{abstract}

\keywords{Quasars --- Sky surveys --- Statistical methods --- Surveys --- Catalogs}

\section{1. Introduction}

The cosmological principle predicts that, aside from the kinematic dipole induced by the Solar
System motion, sufficiently large-scale tracers should be statistically isotropic.
Secrest et al.\ (2022) reported a dipole in the number counts of CatWISE AGN candidates with an
amplitude larger than the naive kinematic expectation, motivating discussion of either new physics
or unmodeled survey systematics.
Because number-count dipoles in flux-limited samples are generically sensitive to completeness and
photometric selection at the faint limit, a decisive diagnostic is whether the signal depends on
the catalog magnitude boundary in a way consistent with selection physics.

This letter presents such a diagnostic using the released CatWISE sample and demonstrates that the
measured dipole exhibits the telltale scaling expected from a small, direction-dependent modulation
of the effective faint threshold.

\section{2. Data and Baseline Dipole Reproduction}

I use the publicly released accepted CatWISE AGN catalog (CatWISE2020; \citealt{catwise2020}) employed by Secrest et al.\ (2022)
(downloaded from their Zenodo release). Following the published baseline cuts, I impose
$|b|>30^\circ$ and require sufficient WISE coverage ($W1{\rm cov}\ge 80$), together with a hard
faint selection $W1\le W1_{\max}$.

Using a simple vector-sum estimator on the unit sphere, I recover a dipole direction consistent
with the published axis. The exact amplitude differs at the few-$\sigma$ level from the published
value, consistent with estimator and weighting differences; the results below focus on
\emph{differential} behavior as a function of the faint-limit selection.

\section{3. The Faint-Limit Scaling Diagnostic}

Let $N(m_{\max})$ denote the number of selected sources under a hard faint cut $m\le m_{\max}$.
If the effective threshold is modulated by a small sky-dependent shift
$m_{\max}\rightarrow m_{\max}+\delta m(\hat{n})$, then to first order the fractional count change is
\begin{equation}
\frac{\delta N}{N}\approx \left.\frac{\mathrm{d}\ln N}{\mathrm{d}m}\right|_{m_{\max}}\delta m(\hat{n})
\equiv \alpha_{\rm edge}(m_{\max})\,\delta m(\hat{n}).
\end{equation}
Thus a selection-driven dipole component should scale with the faint-end slope
$\alpha_{\rm edge}$, whereas a true cosmological dipole would not be expected to track
$\alpha_{\rm edge}$ over small changes in $m_{\max}$.

\subsection{3.1. Empirical behavior across $W1_{\max}$}

I measure the dipole vector $\vec d_{\rm obs}(W1_{\max})$ for 21 values of $W1_{\max}$ from 15.6 to 16.6
in steps of 0.05. For each cut I estimate $\alpha_{\rm edge}$ from the catalog by finite differencing
the counts in narrow bins adjacent to $W1_{\max}$.

Figure~\ref{fig:scaling} shows (i) the dipole amplitude as a function of $W1_{\max}$, (ii) the
corresponding $\alpha_{\rm edge}$, and (iii) the projection of $\vec d_{\rm obs}$ onto the fixed Secrest
axis versus $\alpha_{\rm edge}$.
The measured dipole responds coherently to the faint-limit choice and exhibits approximately linear
behavior in $\alpha_{\rm edge}$, consistent with a selection-driven component.

\subsection{3.2. Two-component scaling fit}

To quantify this, I fit a fixed-axis scaling model:
\begin{equation}
\vec d_{\rm obs}(W1_{\max}) \approx \vec d_0 + \alpha_{\rm edge}(W1_{\max})\,\delta m_{\rm amp}\,\hat{a},
\end{equation}
where $\hat{a}$ is the unit vector along the reported dipole axis, $\delta m_{\rm amp}$ is an effective
dipolar modulation amplitude of the faint limit (in magnitudes), and $\vec d_0$ is a non-scaling residual
term absorbing systematics that do not follow the faint-limit derivative response.

The best fit yields:
\begin{equation}
\delta m_{\rm amp}=(-0.0125\pm0.0027)\,\mathrm{mag}, \qquad \chi^2/\mathrm{dof}\simeq 1.10,
\end{equation}
demonstrating that the dominant variation with $W1_{\max}$ is consistent with the expected selection scaling.

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{fixed_axis_scaling_fit.png}
\caption{Faint-limit scaling diagnostic for the CatWISE quasar dipole.
Left: measured dipole amplitude versus faint magnitude limit $W1_{\max}$.
Middle: faint-edge slope $\alpha_{\rm edge}=\mathrm{d}\ln N/\mathrm{d}m$ estimated at $W1_{\max}$.
Right: projection of the observed dipole vector onto the fixed dipole axis versus $\alpha_{\rm edge}$,
with the best-fit two-component scaling model overplotted. This derivative-linked behavior is the
expected signature of a selection/completeness modulation coupled to the hard faint cut.}
\label{fig:scaling}
\end{figure*}

\section{4. Robustness: Sensitivity to Coverage/Depth Modeling}

If the dipole is selection dominated, its direction and amplitude should be sensitive to how scan-depth
and completeness are modeled. This is observed in two ways:
(i) tightening the catalog coverage cut rapidly produces large apparent dipoles as the footprint collapses
(an effect visible in \texttt{dipole\_vs\_w1covmin.png}), and
(ii) in a cross-validated Poisson generalized linear model fit of HEALPix counts, replacing a catalog-derived
coverage proxy ($W1{\rm cov}$) with an independent imaging-derived depth proxy (unWISE W1 exposure counts)
changes the inferred ``cleaned'' dipole direction substantially.
This behavior is inconsistent with a stable, survey-independent cosmological dipole and instead supports
scan-depth/completeness systematics as the dominant driver.

\section{5. Conclusions}

The CatWISE quasar dipole reported by Secrest et al.\ (2022) exhibits strong dependence on the faint selection
boundary in a manner quantitatively consistent with the derivative response expected for selection/completeness
effects. A two-component scaling model implies an effective dipolar modulation of the faint limit at the
$\sim$10 millimagnitude level along the dipole axis. The direction of ``cleaned'' dipole estimates is also
model dependent under plausible changes in depth templates. These results strongly favor an instrumental or
selection origin for the CatWISE dipole and motivate treating the raw dipole amplitude as a survey-systematics
diagnostic unless an end-to-end completeness model demonstrates otherwise.

\section*{AI assistance disclosure}
The author used AI assistance extensively throughout this project (including for brainstorming, drafting and
editing text, and software development). The numerical results reported here are produced by the accompanying
analysis code and data products (not generated by AI tools).

\section*{Data and code availability}
This analysis uses publicly released CatWISE/Secrest products and publicly released unWISE depth maps. The code,
configuration, and small derived JSON summaries used to produce the figures and numerical results are included
in the accompanying analysis repository (see \texttt{Q\_D\_RES/}).

%% --------------------------------------------------------------------------
%% Inline bibliography (keeps the build self-contained)
%% --------------------------------------------------------------------------
\begin{thebibliography}{}

\bibitem[Secrest et al.(2022)]{secrest2022}
Secrest, N.~J., von Hausegger, S., Rameez, M., Mohayaee, R., \& Sarkar, S.\ 2022, \apjl, 937, L31

%% Optional: add CatWISE/unWISE citations as needed
\bibitem[Marocco et al.(2021)]{catwise2020}
Marocco, F., et al.\ 2021, \apjs, 253, 8, \doi{10.3847/1538-4365/abd805}

\bibitem[Lang(2014)]{unwise_lang2014}
Lang, D.\ 2014, \aj, 147, 108, \doi{10.1088/0004-6256/147/5/108}

\bibitem[Meisner et al.(2017)]{unwise_meisner2017}
Meisner, A.~M., Lang, D., \& Schlegel, D.~J.\ 2017, \aj, 153, 38, \doi{10.3847/1538-3881/153/1/38}

\end{thebibliography}

\end{document}
```

---

## Notes for your submission

- The draft deliberately avoids claiming we uniquely identified the physical origin of the selection gradient
  (scan strategy vs calibration vs masking). The strongest defensible claim is: **the dipole scales like a
  selection/completeness effect at the faint limit**.
- If you want to harden it further, add an Appendix figure showing `dipole_vs_w1covmin.png` (footprint sensitivity)
  and/or `glm_cv_axes_nexp_offset.png` (direction instability under independent depth).
