# Mathematical Model of DJ-Set Constituent Alignment

This note models a DJ set as two independently aligned source-separation lanes:

```text
mix instrumental  <->  constituent instrumental
mix vocals        <->  constituent vocals
```

The two lanes may appear at different times. They are not forced to overlap or
corroborate one another. Tracklist information labels and organizes audio-backed
paths after each audio lane is decoded.

## 1. Reference library

Let \(i \in \{1,\ldots,N\}\) index candidate recordings.

Each recording has two source signals:

\[
x_i^{(I)}(s)
\]

for its instrumental stem, and

\[
x_i^{(V)}(s)
\]

for its vocal stem, where \(s\) is reference-recording time.

The strict routing rule is:

\[
y^{(I)} \leftrightarrow x_i^{(I)}
\qquad\text{and}\qquad
y^{(V)} \leftrightarrow x_i^{(V)}.
\]

Cross-stem comparisons are forbidden:

\[
y^{(I)} \not\leftrightarrow x_i^{(V)},
\qquad
y^{(V)} \not\leftrightarrow x_i^{(I)}.
\]

## 2. DJ-mix observation model

Let \(c \in \{I,V\}\) identify the instrumental or vocal channel.

The separated mix channel is:

\[
y^{(c)}(t)
=
\sum_{i=1}^{N}
a_i^{(c)}(t)\,
\mathcal{T}_i^{(c)}(t)
\left[
x_i^{(c)}\!\left(\phi_i^{(c)}(t)\right)
\right]
+
\epsilon^{(c)}(t).
\]

Here:

- \(t\) is time in the DJ mix.
- \(a_i^{(c)}(t)\geq 0\) is the constituent's time-varying gain.
- \(\phi_i^{(c)}(t)\) maps mix time into reference time.
- \(\mathcal{T}_i^{(c)}(t)\) represents EQ, filtering, reverb, distortion,
  pitch processing, and related effects.
- \(\epsilon^{(c)}(t)\) represents separation leakage, noise, FX, and missing
  references.

The two observation equations are:

\[
y^{(I)}(t)
=
\sum_i
a_i^{(I)}(t)\,
\mathcal{T}_i^{(I)}(t)
\left[
x_i^{(I)}\!\left(\phi_i^{(I)}(t)\right)
\right]
+
\epsilon^{(I)}(t),
\]

\[
y^{(V)}(t)
=
\sum_i
a_i^{(V)}(t)\,
\mathcal{T}_i^{(V)}(t)
\left[
x_i^{(V)}\!\left(\phi_i^{(V)}(t)\right)
\right]
+
\epsilon^{(V)}(t).
\]

In general:

\[
\phi_i^{(I)}(t) \neq \phi_i^{(V)}(t).
\]

A DJ may use a recording's instrumental at one point and its vocal somewhere
else.

## 3. Activity and boundaries

Define binary activity:

\[
z_i^{(c)}(t)
=
\mathbf{1}\!\left[a_i^{(c)}(t)>\tau_a\right].
\]

A constituent segment is a maximal interval \([t_0,t_1)\) such that:

\[
z_i^{(c)}(t)=1
\qquad
\forall t\in[t_0,t_1).
\]

The desired output is:

\[
\mathcal{S}
=
\left\{
(i,c,t_0,t_1,s_0,s_1,\rho,q)
\right\},
\]

where:

- \(i\) is the recording identity;
- \(c\) is the instrumental or vocal lane;
- \([t_0,t_1)\) is the mix-time interval;
- \([s_0,s_1)\) is the reference-time interval;
- \(\rho\) is the playback slope or tempo ratio;
- \(q\) is confidence or evidence.

## 4. Piecewise reference-time mapping

Within one continuous segment:

\[
\phi_i^{(c)}(t)
=
\rho_{ik}^{(c)}t+b_{ik}^{(c)}
\qquad
t\in[t_{ik}^{0},t_{ik}^{1}).
\]

Equivalently:

\[
s=\rho t+b.
\]

The intercept is:

\[
b=s-\rho t.
\]

When playback speed is unchanged:

\[
\rho=1,
\]

and alignment reduces to translation:

\[
s=t+b.
\]

A complete constituent can contain several mappings:

\[
\phi_i^{(c)}(t)
=
\begin{cases}
\rho_{i1}^{(c)}t+b_{i1}^{(c)},
& t\in[t_{i1}^{0},t_{i1}^{1}),\\[4pt]
\varnothing,
& t\in[t_{i1}^{1},t_{i2}^{0}),\\[4pt]
\rho_{i2}^{(c)}t+b_{i2}^{(c)},
& t\in[t_{i2}^{0},t_{i2}^{1}),\\
\vdots
\end{cases}
\]

where \(\varnothing\) means the constituent is inactive. This represents
starts, stops, loops, reference jumps, tempo changes, and later re-entries.

## 5. Fingerprint observations

For one channel \(c\), let:

\[
F_y^{(c)}
=
\{(h_m,t_m)\}
\]

be the mix landmarks, and:

\[
F_i^{(c)}
=
\{(h_r,s_r)\}
\]

be the reference landmarks.

A raw correspondence exists when:

\[
h_m=h_r.
\]

Each match produces:

\[
p_j=(t_j,s_j,w_j).
\]

Common hashes receive less weight:

\[
w_j
=
\frac{1}
{\log_2\!\left(2+n_y(h_j)n_i(h_j)\right)},
\]

where:

\[
n_y(h)=\text{number of mix occurrences of }h,
\]

\[
n_i(h)=\text{number of reference occurrences of }h.
\]

Extremely common hashes abstain:

\[
n_y(h)n_i(h)>C_{\max}
\quad\Longrightarrow\quad
h\text{ is discarded}.
\]

For the vocal lane, the same correspondence representation can be populated by
HuBERT, phonetic, or lyric anchors instead of sparse spectral landmarks:

\[
p_j^{(V)}=(t_j^{(V)},s_j^{(V)},w_j^{(V)}).
\]

The routing remains vocal-to-vocal regardless of the feature representation.

## 6. Diagonal detection

For a candidate playback slope \(\rho\), correct correspondences satisfy:

\[
s_j \approx \rho t_j+b.
\]

Transform each match into intercept space:

\[
b_j=s_j-\rho t_j.
\]

A weighted Hough score is:

\[
H_{\rho}(b)
=
\sum_j
w_j
K_b\!\left(s_j-\rho t_j-b\right),
\]

where \(K_b\) is a narrow kernel or histogram bin.

Candidate diagonals are local maxima:

\[
\mathcal{D}_{\rho}
=
\operatorname{TopK}_b H_{\rho}(b).
\]

## 7. Time-local path support

For diagonal \(d=(\rho,b)\), define:

\[
E_d(t)
=
\sum_j
w_j
K_t(t-t_j)
K_b(s_j-\rho t_j-b).
\]

This measures support near mix time \(t\), rather than counting votes anywhere
in the entire set.

Subtract the collision background:

\[
\widetilde{E}_d(t)
=
\max\left(E_d(t)-B(t),0\right),
\]

where:

\[
B(t)
=
\operatorname{median}_{b'\in\mathcal{B}_{\mathrm{noise}}}
E_{(\rho,b')}(t).
\]

## 8. NULL-aware dynamic programming

For one recording and one stem lane, define states:

\[
\mathcal{Q}
=
\{\mathrm{NULL},d_1,d_2,\ldots,d_K\}.
\]

At discrete mix times \(t_n\), choose state \(q_n\). The optimal path is:

\[
q_{1:T}^{*}
=
\arg\max_{q_{1:T}}
\left[
\sum_{n=1}^{T}e(q_n,t_n)
-
\sum_{n=2}^{T}\lambda(q_{n-1},q_n)
\right].
\]

Emissions are:

\[
e(q,t)
=
\begin{cases}
\widetilde{E}_q(t), & q\neq\mathrm{NULL},\\[4pt]
\eta, & q=\mathrm{NULL}.
\end{cases}
\]

A basic transition cost is:

\[
\lambda(q',q)
=
\begin{cases}
0, & q'=q,\\
\lambda_{\mathrm{enter}},
& q'=\mathrm{NULL},\ q\neq\mathrm{NULL},\\
\lambda_{\mathrm{exit}},
& q'\neq\mathrm{NULL},\ q=\mathrm{NULL},\\
\lambda_{\mathrm{jump}},
& q'\neq q,\ q',q\neq\mathrm{NULL}.
\end{cases}
\]

The NULL state lets the decoder conclude that the constituent is not audible.
Without it, the model must select a least-bad false diagonal everywhere.

## 9. Segment extraction

Each maximal non-NULL run:

\[
q_n=d_k
\qquad
n\in[a,b]
\]

becomes:

\[
t_0=t_a,
\qquad
t_1=t_{b+1},
\]

\[
s_0=\rho_k t_0+b_k,
\qquad
s_1=\rho_k t_1+b_k.
\]

Therefore:

\[
S_{ik}^{(c)}
=
(i,c,t_0,t_1,s_0,s_1,\rho_k).
\]

A re-entry is:

\[
d_k,\ldots,\mathrm{NULL},\ldots,d_k.
\]

A reference jump is:

\[
d_k,\ldots,d_{\ell},
\qquad
b_k\neq b_{\ell}.
\]

## 10. Independent instrumental and vocal inference

Run the complete process separately:

\[
\widehat{\mathcal{S}}^{(I)}
=
\operatorname{Decode}
\left(
F_y^{(I)},
\{F_i^{(I)}\}_{i=1}^{N}
\right),
\]

\[
\widehat{\mathcal{S}}^{(V)}
=
\operatorname{Decode}
\left(
G_y^{(V)},
\{G_i^{(V)}\}_{i=1}^{N}
\right).
\]

\(F\) denotes instrumental fingerprint correspondences. \(G\) denotes
vocal-specific HuBERT, phonetic, or lyric correspondences.

The two decoded sets are not forced to agree spatially or temporally:

\[
\widehat{\mathcal{S}}^{(I)}
\quad\text{is independent of}\quad
\widehat{\mathcal{S}}^{(V)}.
\]

Their union is:

\[
\widehat{\mathcal{S}}
=
\widehat{\mathcal{S}}^{(I)}
\cup
\widehat{\mathcal{S}}^{(V)}.
\]

## 11. Tracklist labeling

Let the ordered tracklist slots be:

\[
L=(\ell_1,\ell_2,\ldots,\ell_M).
\]

Each slot contains identity metadata:

\[
\ell_m=(i_m,c_m,\mathrm{metadata}_m).
\]

Assign decoded audio-backed segments to slots:

\[
A^*
=
\arg\max_A
\left[
\sum_{S\in\widehat{\mathcal{S}}}
\operatorname{AudioScore}(S,A(S))
+
\alpha\operatorname{OrderScore}(A,L)
+
\beta\operatorname{MetadataScore}(A,L)
\right].
\]

Unassigned segments are permitted:

\[
A(S)=\varnothing.
\]

Tracklist data can label and corroborate a detected path, but it cannot create
audio evidence.

## 12. Complete optimization view

The overall inference problem is:

\[
\widehat{\Phi},
\widehat{Z},
\widehat{A}
=
\arg\max_{\Phi,Z,A}
\left[
\mathcal{L}_{I}
\left(y^{(I)},\{x_i^{(I)}\},\Phi^{(I)},Z^{(I)}\right)
+
\mathcal{L}_{V}
\left(y^{(V)},\{x_i^{(V)}\},\Phi^{(V)},Z^{(V)}\right)
+
\mathcal{L}_{\mathrm{tracklist}}(A)
-
\Omega(\Phi,Z,A)
\right].
\]

The posterior factorization is:

\[
p(\Phi,Z,A\mid y,X,L)
\propto
p\!\left(y^{(I)}\mid X^{(I)},\Phi^{(I)},Z^{(I)}\right)
p\!\left(y^{(V)}\mid X^{(V)},\Phi^{(V)},Z^{(V)}\right)
p(A\mid\Phi,Z,L).
\]

In compact form:

\[
\boxed{
\text{instrumental evidence}
\times
\text{vocal evidence}
\times
\text{tracklist labeling}
}
\]

The instrumental and vocal likelihoods are inferred independently before
tracklist attribution.

## 13. The three main problems

### Problem 1: Recovering evidence under superposition

The observed channel still contains several simultaneous constituents,
separation leakage, effects, filtering, and noise:

\[
y^{(c)}(t)
=
\sum_i a_i^{(c)}(t)\,
\mathcal{T}_i^{(c)}(t)
\left[x_i^{(c)}(\phi_i^{(c)}(t))\right]
+
\epsilon^{(c)}(t).
\]

The system must generate reliable like-for-like correspondences even when a
constituent is quiet, masked, transformed, or imperfectly separated.

For instrumentals, sparse landmarks can work when distinctive spectral
structure survives. For vocals, phonetic or HuBERT-style representations are
usually more robust than landmark hashes.

### Problem 2: Recovering the piecewise path and its boundaries

One global offset is insufficient. The same constituent can start, stop, jump
inside its reference, loop, change speed, and re-enter:

\[
\phi_i^{(c)}(t)
=
\begin{cases}
\rho_1t+b_1, & t\in[t_0,t_1),\\
\varnothing, & t\in[t_1,t_2),\\
\rho_2t+b_2, & t\in[t_2,t_3).
\end{cases}
\]

The decoder must determine:

- which diagonal is active;
- when it begins;
- when it ends;
- when to enter NULL;
- when a reference jump or re-entry occurs.

This is the dynamic-programming problem.

### Problem 3: Rejecting ambiguity and assigning tracklist labels

Repeated drums, synth patterns, choruses, loops, and common vocal phrases can
produce convincing false paths:

\[
H_{\rho}(b_{\mathrm{false}})
\approx
H_{\rho}(b_{\mathrm{true}}).
\]

The system must distinguish:

- the correct appearance from a repeated section;
- true sustained support from a short collision;
- a missing channel from negative evidence;
- multiple appearances of one recording from different tracklist slots.

Tracklist order and metadata help label and organize surviving audio paths, but
must remain soft constraints:

\[
\text{tracklist evidence cannot replace missing audio evidence}.
\]

These three problems form the complete alignment stack:

```text
1. Observation: recover like-for-like evidence under superposition.
2. Decoding: recover piecewise paths, gaps, starts, ends, and re-entries.
3. Decision: reject ambiguous paths and assign tracklist labels.
```
